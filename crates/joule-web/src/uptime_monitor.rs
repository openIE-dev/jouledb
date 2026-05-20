//! Uptime monitoring: check definitions (HTTP/TCP/ping simulation), check
//! scheduling, status tracking (up/down/degraded), uptime percentage calculation,
//! downtime windows, status page data, and alerting thresholds.

use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use uuid::Uuid;

// ── Types ──

/// Kind of health check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckKind {
    Http {
        url: String,
        method: String,
        expected_status: u16,
    },
    Tcp {
        host: String,
        port: u16,
    },
    Ping {
        host: String,
    },
    Custom {
        name: String,
    },
}

impl CheckKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            CheckKind::Http { .. } => "http",
            CheckKind::Tcp { .. } => "tcp",
            CheckKind::Ping { .. } => "ping",
            CheckKind::Custom { .. } => "custom",
        }
    }

    pub fn http(url: &str) -> Self {
        CheckKind::Http {
            url: url.to_string(),
            method: "GET".to_string(),
            expected_status: 200,
        }
    }

    pub fn http_with_method(url: &str, method: &str, expected_status: u16) -> Self {
        CheckKind::Http {
            url: url.to_string(),
            method: method.to_string(),
            expected_status,
        }
    }

    pub fn tcp(host: &str, port: u16) -> Self {
        CheckKind::Tcp {
            host: host.to_string(),
            port,
        }
    }

    pub fn ping(host: &str) -> Self {
        CheckKind::Ping {
            host: host.to_string(),
        }
    }
}

/// Current status of a monitored endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointStatus {
    Up,
    Down,
    Degraded,
    Unknown,
}

impl EndpointStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            EndpointStatus::Up => "up",
            EndpointStatus::Down => "down",
            EndpointStatus::Degraded => "degraded",
            EndpointStatus::Unknown => "unknown",
        }
    }

    pub fn is_healthy(&self) -> bool {
        *self == EndpointStatus::Up
    }
}

/// Result of a single check execution.
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub timestamp: DateTime<Utc>,
    pub status: EndpointStatus,
    pub response_time_ms: Option<f64>,
    pub status_code: Option<u16>,
    pub error_message: Option<String>,
}

impl CheckResult {
    pub fn up(response_time_ms: f64) -> Self {
        Self {
            timestamp: Utc::now(),
            status: EndpointStatus::Up,
            response_time_ms: Some(response_time_ms),
            status_code: None,
            error_message: None,
        }
    }

    pub fn up_with_status(response_time_ms: f64, status_code: u16) -> Self {
        Self {
            timestamp: Utc::now(),
            status: EndpointStatus::Up,
            response_time_ms: Some(response_time_ms),
            status_code: Some(status_code),
            error_message: None,
        }
    }

    pub fn degraded(response_time_ms: f64) -> Self {
        Self {
            timestamp: Utc::now(),
            status: EndpointStatus::Degraded,
            response_time_ms: Some(response_time_ms),
            status_code: None,
            error_message: None,
        }
    }

    pub fn down(error: &str) -> Self {
        Self {
            timestamp: Utc::now(),
            status: EndpointStatus::Down,
            response_time_ms: None,
            status_code: None,
            error_message: Some(error.to_string()),
        }
    }

    pub fn down_with_status(status_code: u16, error: &str) -> Self {
        Self {
            timestamp: Utc::now(),
            status: EndpointStatus::Down,
            response_time_ms: None,
            status_code: Some(status_code),
            error_message: Some(error.to_string()),
        }
    }
}

/// Downtime window — a period when the endpoint was down.
#[derive(Debug, Clone)]
pub struct DowntimeWindow {
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub reason: Option<String>,
}

impl DowntimeWindow {
    pub fn new(started_at: DateTime<Utc>) -> Self {
        Self {
            started_at,
            ended_at: None,
            reason: None,
        }
    }

    pub fn with_reason(mut self, reason: &str) -> Self {
        self.reason = Some(reason.to_string());
        self
    }

    pub fn close(&mut self) {
        self.ended_at = Some(Utc::now());
    }

    pub fn close_at(&mut self, time: DateTime<Utc>) {
        self.ended_at = Some(time);
    }

    pub fn duration(&self) -> Duration {
        let end = self.ended_at.unwrap_or_else(Utc::now);
        end - self.started_at
    }

    pub fn is_open(&self) -> bool {
        self.ended_at.is_none()
    }
}

/// Alert threshold configuration for a check.
#[derive(Debug, Clone)]
pub struct AlertConfig {
    pub consecutive_failures: u32,
    pub degraded_latency_ms: f64,
    pub notification_channels: Vec<String>,
    pub cooldown_minutes: u64,
    pub last_alert_at: Option<DateTime<Utc>>,
}

impl AlertConfig {
    pub fn new(consecutive_failures: u32) -> Self {
        Self {
            consecutive_failures,
            degraded_latency_ms: 1000.0,
            notification_channels: Vec::new(),
            cooldown_minutes: 5,
            last_alert_at: None,
        }
    }

    pub fn with_degraded_latency(mut self, ms: f64) -> Self {
        self.degraded_latency_ms = ms;
        self
    }

    pub fn add_channel(&mut self, channel: &str) {
        self.notification_channels.push(channel.to_string());
    }

    pub fn can_alert(&self) -> bool {
        match self.last_alert_at {
            Some(last) => {
                let cooldown = Duration::minutes(self.cooldown_minutes as i64);
                Utc::now() - last > cooldown
            }
            None => true,
        }
    }
}

// ── Uptime Check ──

/// A single uptime check definition.
#[derive(Debug)]
pub struct UptimeCheck {
    pub id: String,
    pub name: String,
    pub check_kind: CheckKind,
    pub interval_secs: u64,
    pub timeout_ms: u64,
    pub current_status: EndpointStatus,
    pub results: Vec<CheckResult>,
    pub downtime_windows: Vec<DowntimeWindow>,
    pub alert_config: AlertConfig,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
    pub enabled: bool,
    pub tags: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
}

impl UptimeCheck {
    pub fn new(name: &str, check_kind: CheckKind) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            check_kind,
            interval_secs: 60,
            timeout_ms: 10_000,
            current_status: EndpointStatus::Unknown,
            results: Vec::new(),
            downtime_windows: Vec::new(),
            alert_config: AlertConfig::new(3),
            consecutive_failures: 0,
            consecutive_successes: 0,
            enabled: true,
            tags: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    pub fn with_interval(mut self, secs: u64) -> Self {
        self.interval_secs = secs;
        self
    }

    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    pub fn with_alert_config(mut self, config: AlertConfig) -> Self {
        self.alert_config = config;
        self
    }

    pub fn set_tag(&mut self, key: &str, value: &str) {
        self.tags.insert(key.to_string(), value.to_string());
    }

    /// Record a check result and update internal state.
    pub fn record_result(&mut self, result: CheckResult) -> Option<Alert> {
        let old_status = self.current_status;

        // Update consecutive counters
        match result.status {
            EndpointStatus::Up => {
                self.consecutive_successes += 1;
                self.consecutive_failures = 0;
            }
            EndpointStatus::Down => {
                self.consecutive_failures += 1;
                self.consecutive_successes = 0;
            }
            EndpointStatus::Degraded => {
                self.consecutive_successes = 0;
                // Degraded counts as partial failure
            }
            EndpointStatus::Unknown => {}
        }

        // Determine latency-based degradation
        let effective_status = if result.status == EndpointStatus::Up {
            if let Some(rt) = result.response_time_ms {
                if rt > self.alert_config.degraded_latency_ms {
                    EndpointStatus::Degraded
                } else {
                    EndpointStatus::Up
                }
            } else {
                EndpointStatus::Up
            }
        } else {
            result.status
        };

        self.current_status = effective_status;

        // Track downtime
        if effective_status == EndpointStatus::Down && old_status != EndpointStatus::Down {
            let reason = result.error_message.clone();
            let mut dw = DowntimeWindow::new(result.timestamp);
            if let Some(r) = reason {
                dw.reason = Some(r);
            }
            self.downtime_windows.push(dw);
        } else if effective_status != EndpointStatus::Down && old_status == EndpointStatus::Down {
            // Close the open downtime window
            if let Some(dw) = self.downtime_windows.last_mut() {
                if dw.is_open() {
                    dw.close_at(result.timestamp);
                }
            }
        }

        self.results.push(result);

        // Check if alert should fire
        if self.consecutive_failures >= self.alert_config.consecutive_failures
            && self.alert_config.can_alert()
        {
            self.alert_config.last_alert_at = Some(Utc::now());
            return Some(Alert {
                check_name: self.name.clone(),
                status: self.current_status,
                consecutive_failures: self.consecutive_failures,
                message: format!(
                    "{} is {} after {} consecutive failures",
                    self.name,
                    self.current_status.as_str(),
                    self.consecutive_failures
                ),
                timestamp: Utc::now(),
            });
        }

        None
    }

    /// Calculate uptime percentage from results.
    pub fn uptime_percent(&self) -> f64 {
        if self.results.is_empty() {
            return 100.0;
        }
        let up_count = self
            .results
            .iter()
            .filter(|r| r.status == EndpointStatus::Up)
            .count();
        (up_count as f64 / self.results.len() as f64) * 100.0
    }

    /// Uptime percentage over a rolling window.
    pub fn uptime_percent_window(&self, window: usize) -> f64 {
        let start = self.results.len().saturating_sub(window);
        let slice = &self.results[start..];
        if slice.is_empty() {
            return 100.0;
        }
        let up_count = slice
            .iter()
            .filter(|r| r.status == EndpointStatus::Up)
            .count();
        (up_count as f64 / slice.len() as f64) * 100.0
    }

    /// Average response time in ms.
    pub fn avg_response_time_ms(&self) -> Option<f64> {
        let times: Vec<f64> = self
            .results
            .iter()
            .filter_map(|r| r.response_time_ms)
            .collect();
        if times.is_empty() {
            return None;
        }
        Some(times.iter().sum::<f64>() / times.len() as f64)
    }

    /// P99 response time.
    pub fn p99_response_time_ms(&self) -> Option<f64> {
        let mut times: Vec<f64> = self
            .results
            .iter()
            .filter_map(|r| r.response_time_ms)
            .collect();
        if times.is_empty() {
            return None;
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((times.len() as f64) * 0.99).ceil() as usize;
        let idx = idx.min(times.len()) - 1;
        Some(times[idx])
    }

    /// Total downtime duration.
    pub fn total_downtime(&self) -> Duration {
        self.downtime_windows
            .iter()
            .fold(Duration::zero(), |acc, dw| acc + dw.duration())
    }

    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    pub fn downtime_window_count(&self) -> usize {
        self.downtime_windows.len()
    }
}

/// An alert fired by a check.
#[derive(Debug, Clone)]
pub struct Alert {
    pub check_name: String,
    pub status: EndpointStatus,
    pub consecutive_failures: u32,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

// ── Status Page ──

/// Data for a status page component.
#[derive(Debug, Clone)]
pub struct StatusPageComponent {
    pub name: String,
    pub status: EndpointStatus,
    pub uptime_30d: f64,
    pub avg_response_ms: Option<f64>,
    pub last_checked: Option<DateTime<Utc>>,
    pub description: Option<String>,
}

/// Aggregate data for a status page.
#[derive(Debug)]
pub struct StatusPage {
    pub title: String,
    pub overall_status: EndpointStatus,
    pub components: Vec<StatusPageComponent>,
    pub updated_at: DateTime<Utc>,
}

impl StatusPage {
    pub fn from_checks(title: &str, checks: &[&UptimeCheck]) -> Self {
        let components: Vec<StatusPageComponent> = checks
            .iter()
            .map(|check| {
                let last_checked = check.results.last().map(|r| r.timestamp);
                StatusPageComponent {
                    name: check.name.clone(),
                    status: check.current_status,
                    uptime_30d: check.uptime_percent(),
                    avg_response_ms: check.avg_response_time_ms(),
                    last_checked,
                    description: None,
                }
            })
            .collect();

        let overall = if components.iter().any(|c| c.status == EndpointStatus::Down) {
            EndpointStatus::Down
        } else if components
            .iter()
            .any(|c| c.status == EndpointStatus::Degraded)
        {
            EndpointStatus::Degraded
        } else if components.iter().all(|c| c.status == EndpointStatus::Up) {
            EndpointStatus::Up
        } else {
            EndpointStatus::Unknown
        };

        Self {
            title: title.to_string(),
            overall_status: overall,
            components,
            updated_at: Utc::now(),
        }
    }

    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    pub fn healthy_count(&self) -> usize {
        self.components
            .iter()
            .filter(|c| c.status == EndpointStatus::Up)
            .count()
    }

    pub fn unhealthy_count(&self) -> usize {
        self.components
            .iter()
            .filter(|c| c.status == EndpointStatus::Down)
            .count()
    }
}

// ── Monitor ──

/// Uptime monitor managing multiple checks.
#[derive(Debug, Default)]
pub struct UptimeMonitor {
    checks: Vec<UptimeCheck>,
}

impl UptimeMonitor {
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
        }
    }

    pub fn add_check(&mut self, check: UptimeCheck) -> String {
        let id = check.id.clone();
        self.checks.push(check);
        id
    }

    pub fn get(&self, id: &str) -> Option<&UptimeCheck> {
        self.checks.iter().find(|c| c.id == id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut UptimeCheck> {
        self.checks.iter_mut().find(|c| c.id == id)
    }

    pub fn get_by_name(&self, name: &str) -> Option<&UptimeCheck> {
        self.checks.iter().find(|c| c.name == name)
    }

    pub fn check_count(&self) -> usize {
        self.checks.len()
    }

    pub fn enabled_checks(&self) -> Vec<&UptimeCheck> {
        self.checks.iter().filter(|c| c.enabled).collect()
    }

    /// Get checks that are due for execution based on interval.
    pub fn due_checks(&self, now: DateTime<Utc>) -> Vec<&UptimeCheck> {
        self.checks
            .iter()
            .filter(|c| {
                if !c.enabled {
                    return false;
                }
                match c.results.last() {
                    Some(last) => {
                        let elapsed = (now - last.timestamp).num_seconds();
                        elapsed >= c.interval_secs as i64
                    }
                    None => true, // Never checked
                }
            })
            .collect()
    }

    /// Build a status page from all checks.
    pub fn status_page(&self, title: &str) -> StatusPage {
        let check_refs: Vec<&UptimeCheck> = self.checks.iter().collect();
        StatusPage::from_checks(title, &check_refs)
    }

    /// Overall system uptime (worst case across all checks).
    pub fn overall_uptime_percent(&self) -> f64 {
        if self.checks.is_empty() {
            return 100.0;
        }
        self.checks
            .iter()
            .map(|c| c.uptime_percent())
            .fold(f64::INFINITY, f64::min)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn make_http_check(name: &str) -> UptimeCheck {
        UptimeCheck::new(name, CheckKind::http("https://example.com/health"))
    }

    #[test]
    fn test_check_creation() {
        let check = make_http_check("api");
        assert_eq!(check.name, "api");
        assert_eq!(check.current_status, EndpointStatus::Unknown);
        assert!(check.enabled);
        assert_eq!(check.check_kind.as_str(), "http");
    }

    #[test]
    fn test_check_kinds() {
        assert_eq!(CheckKind::http("http://example.com").as_str(), "http");
        assert_eq!(CheckKind::tcp("db-host", 5432).as_str(), "tcp");
        assert_eq!(CheckKind::ping("host").as_str(), "ping");
        assert_eq!(
            CheckKind::Custom {
                name: "x".to_string()
            }
            .as_str(),
            "custom"
        );
    }

    #[test]
    fn test_record_up() {
        let mut check = make_http_check("api");
        let alert = check.record_result(CheckResult::up(50.0));
        assert!(alert.is_none());
        assert_eq!(check.current_status, EndpointStatus::Up);
        assert_eq!(check.consecutive_successes, 1);
    }

    #[test]
    fn test_record_down() {
        let mut check = make_http_check("api");
        check.record_result(CheckResult::down("Connection refused"));
        assert_eq!(check.current_status, EndpointStatus::Down);
        assert_eq!(check.consecutive_failures, 1);
        assert_eq!(check.downtime_window_count(), 1);
    }

    #[test]
    fn test_alert_on_consecutive_failures() {
        let mut check =
            make_http_check("api").with_alert_config(AlertConfig::new(3));
        check.record_result(CheckResult::down("err"));
        check.record_result(CheckResult::down("err"));
        let alert = check.record_result(CheckResult::down("err"));
        assert!(alert.is_some());
        assert_eq!(alert.unwrap().consecutive_failures, 3);
    }

    #[test]
    fn test_uptime_percent() {
        let mut check = make_http_check("api");
        for _ in 0..9 {
            check.record_result(CheckResult::up(50.0));
        }
        check.record_result(CheckResult::down("err"));
        assert!((check.uptime_percent() - 90.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_uptime_percent_window() {
        let mut check = make_http_check("api");
        for _ in 0..5 {
            check.record_result(CheckResult::down("err"));
        }
        for _ in 0..5 {
            check.record_result(CheckResult::up(50.0));
        }
        // Last 5 results: all up
        assert!((check.uptime_percent_window(5) - 100.0).abs() < f64::EPSILON);
        // All 10: 50%
        assert!((check.uptime_percent_window(10) - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg_response_time() {
        let mut check = make_http_check("api");
        check.record_result(CheckResult::up(100.0));
        check.record_result(CheckResult::up(200.0));
        check.record_result(CheckResult::up(300.0));
        let avg = check.avg_response_time_ms().unwrap();
        assert!((avg - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_p99_response_time() {
        let mut check = make_http_check("api");
        for i in 1..=100 {
            check.record_result(CheckResult::up(i as f64));
        }
        let p99 = check.p99_response_time_ms().unwrap();
        assert!(p99 >= 99.0);
    }

    #[test]
    fn test_downtime_window() {
        let mut check = make_http_check("api");
        check.record_result(CheckResult::up(50.0));
        check.record_result(CheckResult::down("error"));
        check.record_result(CheckResult::down("error"));
        check.record_result(CheckResult::up(50.0));
        assert_eq!(check.downtime_window_count(), 1);
        assert!(!check.downtime_windows[0].is_open());
    }

    #[test]
    fn test_downtime_window_still_open() {
        let mut check = make_http_check("api");
        check.record_result(CheckResult::down("error"));
        assert_eq!(check.downtime_window_count(), 1);
        assert!(check.downtime_windows[0].is_open());
    }

    #[test]
    fn test_degraded_by_latency() {
        let mut check = make_http_check("api").with_alert_config(
            AlertConfig::new(3).with_degraded_latency(100.0),
        );
        check.record_result(CheckResult::up(500.0));
        assert_eq!(check.current_status, EndpointStatus::Degraded);
    }

    #[test]
    fn test_monitor_add_and_get() {
        let mut monitor = UptimeMonitor::new();
        let id = monitor.add_check(make_http_check("api"));
        assert_eq!(monitor.check_count(), 1);
        assert!(monitor.get(&id).is_some());
        assert!(monitor.get_by_name("api").is_some());
    }

    #[test]
    fn test_monitor_due_checks() {
        let mut monitor = UptimeMonitor::new();
        let check = make_http_check("api").with_interval(60);
        monitor.add_check(check);
        // No results yet → always due
        let due = monitor.due_checks(Utc::now());
        assert_eq!(due.len(), 1);
    }

    #[test]
    fn test_status_page() {
        let mut monitor = UptimeMonitor::new();
        let id1 = monitor.add_check(make_http_check("api"));
        let id2 = monitor.add_check(make_http_check("web"));

        monitor
            .get_mut(&id1)
            .unwrap()
            .record_result(CheckResult::up(50.0));
        monitor
            .get_mut(&id2)
            .unwrap()
            .record_result(CheckResult::down("err"));

        let page = monitor.status_page("Acme Status");
        assert_eq!(page.component_count(), 2);
        assert_eq!(page.healthy_count(), 1);
        assert_eq!(page.unhealthy_count(), 1);
        assert_eq!(page.overall_status, EndpointStatus::Down);
    }

    #[test]
    fn test_status_page_all_up() {
        let mut monitor = UptimeMonitor::new();
        let id1 = monitor.add_check(make_http_check("api"));
        let id2 = monitor.add_check(make_http_check("web"));
        monitor
            .get_mut(&id1)
            .unwrap()
            .record_result(CheckResult::up(50.0));
        monitor
            .get_mut(&id2)
            .unwrap()
            .record_result(CheckResult::up(60.0));

        let page = monitor.status_page("Status");
        assert_eq!(page.overall_status, EndpointStatus::Up);
    }

    #[test]
    fn test_overall_uptime() {
        let mut monitor = UptimeMonitor::new();
        let id1 = monitor.add_check(make_http_check("api"));
        let id2 = monitor.add_check(make_http_check("web"));

        // api: 100% up
        for _ in 0..10 {
            monitor
                .get_mut(&id1)
                .unwrap()
                .record_result(CheckResult::up(50.0));
        }
        // web: 50% up
        for _ in 0..5 {
            monitor
                .get_mut(&id2)
                .unwrap()
                .record_result(CheckResult::up(50.0));
        }
        for _ in 0..5 {
            monitor
                .get_mut(&id2)
                .unwrap()
                .record_result(CheckResult::down("err"));
        }

        // Overall = min(100, 50) = 50
        assert!((monitor.overall_uptime_percent() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_endpoint_status_healthy() {
        assert!(EndpointStatus::Up.is_healthy());
        assert!(!EndpointStatus::Down.is_healthy());
        assert!(!EndpointStatus::Degraded.is_healthy());
    }

    #[test]
    fn test_alert_config_cooldown() {
        let mut config = AlertConfig::new(3);
        assert!(config.can_alert());
        config.last_alert_at = Some(Utc::now());
        assert!(!config.can_alert()); // Within cooldown
    }

    #[test]
    fn test_check_with_tags() {
        let mut check = make_http_check("api");
        check.set_tag("env", "production");
        assert_eq!(check.tags.get("env").unwrap(), "production");
    }

    #[test]
    fn test_check_result_with_status_code() {
        let result = CheckResult::up_with_status(50.0, 200);
        assert_eq!(result.status_code, Some(200));
        assert_eq!(result.status, EndpointStatus::Up);
    }

    #[test]
    fn test_downtime_window_manual() {
        let mut dw = DowntimeWindow::new(Utc::now()).with_reason("Maintenance");
        assert!(dw.is_open());
        assert_eq!(dw.reason.as_deref(), Some("Maintenance"));
        dw.close();
        assert!(!dw.is_open());
    }
}
