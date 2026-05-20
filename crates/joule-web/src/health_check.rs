//! Health check endpoints for service observability.
//!
//! Provides Kubernetes-style liveness, readiness, and startup probes,
//! dependency health tracking, degraded state detection, and aggregated
//! health status reporting. Pure Rust — no network or filesystem access.

use std::collections::HashMap;
use std::fmt;

// ── Status ────────────────────────────────────────────────────────

/// Overall health status of a component or the service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HealthStatus {
    /// Fully operational.
    Healthy,
    /// Running but with reduced capability.
    Degraded,
    /// Not operational.
    Unhealthy,
    /// Status not yet determined (startup).
    Unknown,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Unhealthy => write!(f, "unhealthy"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

impl HealthStatus {
    /// Merge two statuses, returning the worse of the two.
    pub fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unhealthy, _) | (_, Self::Unhealthy) => Self::Unhealthy,
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::Degraded, _) | (_, Self::Degraded) => Self::Degraded,
            _ => Self::Healthy,
        }
    }
}

// ── Dependency check ──────────────────────────────────────────────

/// Result of checking a single dependency.
#[derive(Debug, Clone)]
pub struct DependencyCheck {
    /// Name of the dependency (e.g. "database", "cache").
    pub name: String,
    /// Health status.
    pub status: HealthStatus,
    /// Optional latency in microseconds.
    pub latency_us: Option<u64>,
    /// Optional human-readable message.
    pub message: Option<String>,
    /// Whether this dependency is critical (affects readiness).
    pub critical: bool,
}

impl DependencyCheck {
    /// Create a healthy dependency check.
    pub fn healthy(name: &str, critical: bool) -> Self {
        Self {
            name: name.to_string(),
            status: HealthStatus::Healthy,
            latency_us: None,
            message: None,
            critical,
        }
    }

    /// Create an unhealthy dependency check.
    pub fn unhealthy(name: &str, critical: bool, msg: &str) -> Self {
        Self {
            name: name.to_string(),
            status: HealthStatus::Unhealthy,
            latency_us: None,
            message: Some(msg.to_string()),
            critical,
        }
    }

    /// Create a degraded dependency check.
    pub fn degraded(name: &str, critical: bool, msg: &str) -> Self {
        Self {
            name: name.to_string(),
            status: HealthStatus::Degraded,
            latency_us: None,
            message: Some(msg.to_string()),
            critical,
        }
    }

    /// Set latency in microseconds.
    pub fn with_latency(mut self, us: u64) -> Self {
        self.latency_us = Some(us);
        self
    }
}

// ── Probe types ───────────────────────────────────────────────────

/// Kubernetes-style probe kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeKind {
    /// Is the process alive? (restart if failing)
    Liveness,
    /// Is the service ready to accept traffic?
    Readiness,
    /// Has the service finished starting up?
    Startup,
}

impl fmt::Display for ProbeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Liveness => write!(f, "liveness"),
            Self::Readiness => write!(f, "readiness"),
            Self::Startup => write!(f, "startup"),
        }
    }
}

/// Outcome of a probe evaluation.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub kind: ProbeKind,
    pub status: HealthStatus,
    pub message: Option<String>,
}

impl ProbeResult {
    /// HTTP-style status code (200 = ok, 503 = failing).
    pub fn http_status_code(&self) -> u16 {
        match self.status {
            HealthStatus::Healthy | HealthStatus::Degraded => 200,
            HealthStatus::Unhealthy | HealthStatus::Unknown => 503,
        }
    }
}

// ── Threshold configuration ───────────────────────────────────────

/// Thresholds that govern when a healthy service transitions to degraded.
#[derive(Debug, Clone)]
pub struct DegradedThresholds {
    /// Maximum acceptable dependency latency (microseconds).
    pub max_latency_us: u64,
    /// Maximum number of non-critical unhealthy dependencies before degraded.
    pub max_unhealthy_non_critical: usize,
    /// Consecutive failures before marking unhealthy.
    pub failure_threshold: u32,
    /// Consecutive successes to recover from unhealthy.
    pub success_threshold: u32,
}

impl Default for DegradedThresholds {
    fn default() -> Self {
        Self {
            max_latency_us: 5_000_000,
            max_unhealthy_non_critical: 1,
            failure_threshold: 3,
            success_threshold: 1,
        }
    }
}

// ── Health registry ───────────────────────────────────────────────

/// Aggregates dependency checks and evaluates probes.
#[derive(Debug, Clone)]
pub struct HealthRegistry {
    checks: Vec<DependencyCheck>,
    startup_complete: bool,
    failure_counts: HashMap<String, u32>,
    success_counts: HashMap<String, u32>,
    thresholds: DegradedThresholds,
    alive: bool,
}

impl HealthRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            startup_complete: false,
            failure_counts: HashMap::new(),
            success_counts: HashMap::new(),
            thresholds: DegradedThresholds::default(),
            alive: true,
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(thresholds: DegradedThresholds) -> Self {
        Self {
            thresholds,
            ..Self::new()
        }
    }

    /// Mark startup as complete.
    pub fn complete_startup(&mut self) {
        self.startup_complete = true;
    }

    /// Mark the process as not alive (liveness probe will fail).
    pub fn mark_not_alive(&mut self) {
        self.alive = false;
    }

    /// Record a dependency check result.
    pub fn record_check(&mut self, check: DependencyCheck) {
        let name = check.name.clone();

        match check.status {
            HealthStatus::Healthy => {
                let sc = self.success_counts.entry(name.clone()).or_insert(0);
                *sc = sc.saturating_add(1);
                self.failure_counts.insert(name, 0);
            }
            HealthStatus::Unhealthy | HealthStatus::Unknown => {
                let fc = self.failure_counts.entry(name.clone()).or_insert(0);
                *fc = fc.saturating_add(1);
                self.success_counts.insert(name, 0);
            }
            HealthStatus::Degraded => {
                self.success_counts.insert(name, 0);
            }
        }

        if let Some(existing) = self.checks.iter_mut().find(|c| c.name == check.name) {
            *existing = check;
        } else {
            self.checks.push(check);
        }
    }

    /// Get all current dependency checks.
    pub fn checks(&self) -> &[DependencyCheck] {
        &self.checks
    }

    /// Evaluate liveness probe.
    pub fn liveness(&self) -> ProbeResult {
        if self.alive {
            ProbeResult {
                kind: ProbeKind::Liveness,
                status: HealthStatus::Healthy,
                message: None,
            }
        } else {
            ProbeResult {
                kind: ProbeKind::Liveness,
                status: HealthStatus::Unhealthy,
                message: Some("process marked not alive".into()),
            }
        }
    }

    /// Evaluate startup probe.
    pub fn startup(&self) -> ProbeResult {
        if self.startup_complete {
            ProbeResult {
                kind: ProbeKind::Startup,
                status: HealthStatus::Healthy,
                message: None,
            }
        } else {
            ProbeResult {
                kind: ProbeKind::Startup,
                status: HealthStatus::Unhealthy,
                message: Some("startup not yet complete".into()),
            }
        }
    }

    /// Evaluate readiness — ready when all critical deps healthy and started.
    pub fn readiness(&self) -> ProbeResult {
        if !self.startup_complete {
            return ProbeResult {
                kind: ProbeKind::Readiness,
                status: HealthStatus::Unhealthy,
                message: Some("startup not yet complete".into()),
            };
        }

        for check in &self.checks {
            if check.critical && check.status == HealthStatus::Unhealthy {
                let fc = self.failure_counts.get(&check.name).copied().unwrap_or(0);
                if fc >= self.thresholds.failure_threshold {
                    return ProbeResult {
                        kind: ProbeKind::Readiness,
                        status: HealthStatus::Unhealthy,
                        message: Some(format!(
                            "critical dependency '{}' unhealthy ({} consecutive failures)",
                            check.name, fc
                        )),
                    };
                }
            }
        }

        ProbeResult {
            kind: ProbeKind::Readiness,
            status: HealthStatus::Healthy,
            message: None,
        }
    }

    /// Compute the aggregate health status considering all dependencies.
    pub fn aggregate_status(&self) -> HealthStatus {
        if !self.alive {
            return HealthStatus::Unhealthy;
        }
        if !self.startup_complete {
            return HealthStatus::Unknown;
        }

        let mut status = HealthStatus::Healthy;
        let mut unhealthy_non_critical = 0usize;

        for check in &self.checks {
            match check.status {
                HealthStatus::Unhealthy => {
                    if check.critical {
                        return HealthStatus::Unhealthy;
                    }
                    unhealthy_non_critical += 1;
                }
                HealthStatus::Degraded => {
                    status = status.merge(HealthStatus::Degraded);
                }
                HealthStatus::Unknown => {
                    if check.critical {
                        status = status.merge(HealthStatus::Unknown);
                    }
                }
                HealthStatus::Healthy => {
                    if let Some(lat) = check.latency_us {
                        if lat > self.thresholds.max_latency_us {
                            status = status.merge(HealthStatus::Degraded);
                        }
                    }
                }
            }
        }

        if unhealthy_non_critical > self.thresholds.max_unhealthy_non_critical {
            status = status.merge(HealthStatus::Degraded);
        }

        status
    }

    /// Build a full health report.
    pub fn report(&self) -> HealthReport {
        let aggregate = self.aggregate_status();
        let liveness = self.liveness();
        let readiness = self.readiness();
        let startup = self.startup();

        let deps: Vec<DependencyReport> = self
            .checks
            .iter()
            .map(|c| DependencyReport {
                name: c.name.clone(),
                status: c.status,
                latency_us: c.latency_us,
                message: c.message.clone(),
                critical: c.critical,
                consecutive_failures: self.failure_counts.get(&c.name).copied().unwrap_or(0),
                consecutive_successes: self.success_counts.get(&c.name).copied().unwrap_or(0),
            })
            .collect();

        HealthReport {
            status: aggregate,
            liveness_status: liveness.status,
            readiness_status: readiness.status,
            startup_status: startup.status,
            dependencies: deps,
        }
    }
}

impl Default for HealthRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Report structures ─────────────────────────────────────────────

/// Per-dependency report entry.
#[derive(Debug, Clone)]
pub struct DependencyReport {
    pub name: String,
    pub status: HealthStatus,
    pub latency_us: Option<u64>,
    pub message: Option<String>,
    pub critical: bool,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
}

/// Full health report.
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub status: HealthStatus,
    pub liveness_status: HealthStatus,
    pub readiness_status: HealthStatus,
    pub startup_status: HealthStatus,
    pub dependencies: Vec<DependencyReport>,
}

impl HealthReport {
    /// HTTP-style status code for the aggregate status.
    pub fn http_status_code(&self) -> u16 {
        match self.status {
            HealthStatus::Healthy | HealthStatus::Degraded => 200,
            _ => 503,
        }
    }

    /// Serialize to a JSON string.
    pub fn to_json(&self) -> String {
        let deps_json: Vec<String> = self
            .dependencies
            .iter()
            .map(|d| {
                let lat = match d.latency_us {
                    Some(v) => v.to_string(),
                    None => "null".to_string(),
                };
                let msg = match &d.message {
                    Some(m) => format!(r##""{}""##, m.replace('"', r#"\""#)),
                    None => "null".to_string(),
                };
                format!(
                    concat!(
                        r#"{{"name":"{}","status":"{}","latency_us":{},"message":{},"#,
                        r#""critical":{},"consecutive_failures":{},"consecutive_successes":{}}}"#,
                    ),
                    d.name, d.status, lat, msg, d.critical,
                    d.consecutive_failures, d.consecutive_successes
                )
            })
            .collect();

        format!(
            concat!(
                r#"{{"status":"{}","liveness":"{}","readiness":"{}","#,
                r#""startup":"{}","dependencies":[{}]}}"#,
            ),
            self.status,
            self.liveness_status,
            self.readiness_status,
            self.startup_status,
            deps_json.join(",")
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "degraded");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "unhealthy");
        assert_eq!(HealthStatus::Unknown.to_string(), "unknown");
    }

    #[test]
    fn status_merge_symmetry() {
        let cases = [
            HealthStatus::Healthy,
            HealthStatus::Degraded,
            HealthStatus::Unhealthy,
            HealthStatus::Unknown,
        ];
        for a in &cases {
            for b in &cases {
                assert_eq!(a.merge(*b), b.merge(*a), "merge({a},{b}) asymmetric");
            }
        }
    }

    #[test]
    fn status_merge_unhealthy_wins() {
        assert_eq!(
            HealthStatus::Healthy.merge(HealthStatus::Unhealthy),
            HealthStatus::Unhealthy
        );
        assert_eq!(
            HealthStatus::Degraded.merge(HealthStatus::Unhealthy),
            HealthStatus::Unhealthy
        );
    }

    #[test]
    fn status_merge_degraded_over_healthy() {
        assert_eq!(
            HealthStatus::Healthy.merge(HealthStatus::Degraded),
            HealthStatus::Degraded
        );
    }

    #[test]
    fn status_merge_unknown_over_degraded() {
        assert_eq!(
            HealthStatus::Degraded.merge(HealthStatus::Unknown),
            HealthStatus::Unknown
        );
    }

    #[test]
    fn dep_check_healthy() {
        let c = DependencyCheck::healthy("db", true);
        assert_eq!(c.status, HealthStatus::Healthy);
        assert!(c.critical);
        assert!(c.latency_us.is_none());
    }

    #[test]
    fn dep_check_unhealthy() {
        let c = DependencyCheck::unhealthy("cache", false, "connection refused");
        assert_eq!(c.status, HealthStatus::Unhealthy);
        assert!(!c.critical);
        assert_eq!(c.message.as_deref(), Some("connection refused"));
    }

    #[test]
    fn dep_check_with_latency() {
        let c = DependencyCheck::healthy("db", true).with_latency(1500);
        assert_eq!(c.latency_us, Some(1500));
    }

    #[test]
    fn probe_kind_display() {
        assert_eq!(ProbeKind::Liveness.to_string(), "liveness");
        assert_eq!(ProbeKind::Readiness.to_string(), "readiness");
        assert_eq!(ProbeKind::Startup.to_string(), "startup");
    }

    #[test]
    fn probe_http_status_healthy() {
        let p = ProbeResult {
            kind: ProbeKind::Liveness,
            status: HealthStatus::Healthy,
            message: None,
        };
        assert_eq!(p.http_status_code(), 200);
    }

    #[test]
    fn probe_http_status_unhealthy() {
        let p = ProbeResult {
            kind: ProbeKind::Readiness,
            status: HealthStatus::Unhealthy,
            message: Some("bad".into()),
        };
        assert_eq!(p.http_status_code(), 503);
    }

    #[test]
    fn registry_new_defaults() {
        let reg = HealthRegistry::new();
        assert!(reg.checks().is_empty());
        assert!(!reg.startup_complete);
        assert!(reg.alive);
    }

    #[test]
    fn liveness_initially_healthy() {
        let reg = HealthRegistry::new();
        let p = reg.liveness();
        assert_eq!(p.status, HealthStatus::Healthy);
        assert_eq!(p.kind, ProbeKind::Liveness);
    }

    #[test]
    fn liveness_after_mark_not_alive() {
        let mut reg = HealthRegistry::new();
        reg.mark_not_alive();
        let p = reg.liveness();
        assert_eq!(p.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn startup_before_complete() {
        let reg = HealthRegistry::new();
        let p = reg.startup();
        assert_eq!(p.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn startup_after_complete() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        let p = reg.startup();
        assert_eq!(p.status, HealthStatus::Healthy);
    }

    #[test]
    fn readiness_fails_before_startup() {
        let reg = HealthRegistry::new();
        let p = reg.readiness();
        assert_eq!(p.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn readiness_healthy_after_startup_no_deps() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        assert_eq!(reg.readiness().status, HealthStatus::Healthy);
    }

    #[test]
    fn readiness_fails_critical_dep_over_threshold() {
        let thresholds = DegradedThresholds {
            failure_threshold: 2,
            ..Default::default()
        };
        let mut reg = HealthRegistry::with_thresholds(thresholds);
        reg.complete_startup();
        reg.record_check(DependencyCheck::unhealthy("db", true, "down"));
        reg.record_check(DependencyCheck::unhealthy("db", true, "down"));
        assert_eq!(reg.readiness().status, HealthStatus::Unhealthy);
    }

    #[test]
    fn readiness_ok_non_critical_unhealthy() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        for _ in 0..10 {
            reg.record_check(DependencyCheck::unhealthy("metrics", false, "timeout"));
        }
        assert_eq!(reg.readiness().status, HealthStatus::Healthy);
    }

    #[test]
    fn aggregate_unknown_before_startup() {
        let reg = HealthRegistry::new();
        assert_eq!(reg.aggregate_status(), HealthStatus::Unknown);
    }

    #[test]
    fn aggregate_healthy_after_startup() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        assert_eq!(reg.aggregate_status(), HealthStatus::Healthy);
    }

    #[test]
    fn aggregate_unhealthy_critical_dep() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        reg.record_check(DependencyCheck::unhealthy("db", true, "gone"));
        assert_eq!(reg.aggregate_status(), HealthStatus::Unhealthy);
    }

    #[test]
    fn aggregate_degraded_from_latency() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        reg.record_check(DependencyCheck::healthy("db", true).with_latency(10_000_000));
        assert_eq!(reg.aggregate_status(), HealthStatus::Degraded);
    }

    #[test]
    fn aggregate_degraded_many_non_critical() {
        let thresholds = DegradedThresholds {
            max_unhealthy_non_critical: 1,
            ..Default::default()
        };
        let mut reg = HealthRegistry::with_thresholds(thresholds);
        reg.complete_startup();
        reg.record_check(DependencyCheck::unhealthy("cache", false, "x"));
        reg.record_check(DependencyCheck::unhealthy("metrics", false, "y"));
        assert_eq!(reg.aggregate_status(), HealthStatus::Degraded);
    }

    #[test]
    fn failure_count_tracks_consecutive() {
        let mut reg = HealthRegistry::new();
        reg.record_check(DependencyCheck::unhealthy("db", true, "a"));
        reg.record_check(DependencyCheck::unhealthy("db", true, "b"));
        reg.record_check(DependencyCheck::unhealthy("db", true, "c"));
        assert_eq!(reg.failure_counts.get("db").copied(), Some(3));
    }

    #[test]
    fn success_resets_failure_count() {
        let mut reg = HealthRegistry::new();
        reg.record_check(DependencyCheck::unhealthy("db", true, "a"));
        reg.record_check(DependencyCheck::unhealthy("db", true, "b"));
        assert_eq!(reg.failure_counts["db"], 2);
        reg.record_check(DependencyCheck::healthy("db", true));
        assert_eq!(reg.failure_counts["db"], 0);
        assert_eq!(reg.success_counts["db"], 1);
    }

    #[test]
    fn record_check_replaces_existing() {
        let mut reg = HealthRegistry::new();
        reg.record_check(DependencyCheck::healthy("db", true));
        assert_eq!(reg.checks().len(), 1);
        reg.record_check(DependencyCheck::unhealthy("db", true, "oops"));
        assert_eq!(reg.checks().len(), 1);
        assert_eq!(reg.checks()[0].status, HealthStatus::Unhealthy);
    }

    #[test]
    fn report_structure() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        reg.record_check(DependencyCheck::healthy("db", true).with_latency(500));
        reg.record_check(DependencyCheck::degraded("cache", false, "slow"));
        let rpt = reg.report();
        assert_eq!(rpt.status, HealthStatus::Degraded);
        assert_eq!(rpt.liveness_status, HealthStatus::Healthy);
        assert_eq!(rpt.readiness_status, HealthStatus::Healthy);
        assert_eq!(rpt.startup_status, HealthStatus::Healthy);
        assert_eq!(rpt.dependencies.len(), 2);
    }

    #[test]
    fn report_json_fields() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        reg.record_check(DependencyCheck::healthy("db", true));
        let json = reg.report().to_json();
        assert!(json.contains(r#""status":"healthy""#));
        assert!(json.contains(r#""liveness":"healthy""#));
        assert!(json.contains(r#""name":"db""#));
    }

    #[test]
    fn report_http_status_codes() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        assert_eq!(reg.report().http_status_code(), 200);

        reg.record_check(DependencyCheck::degraded("x", false, "slow"));
        assert_eq!(reg.report().http_status_code(), 200);

        let reg2 = HealthRegistry::new(); // not started
        assert_eq!(reg2.report().http_status_code(), 503);
    }

    #[test]
    fn degraded_resets_success_count() {
        let mut reg = HealthRegistry::new();
        reg.record_check(DependencyCheck::healthy("x", false));
        reg.record_check(DependencyCheck::healthy("x", false));
        assert_eq!(reg.success_counts["x"], 2);
        reg.record_check(DependencyCheck::degraded("x", false, "slow"));
        assert_eq!(reg.success_counts["x"], 0);
    }

    #[test]
    fn aggregate_unhealthy_when_not_alive() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        reg.mark_not_alive();
        assert_eq!(reg.aggregate_status(), HealthStatus::Unhealthy);
    }

    #[test]
    fn dep_report_consecutive_counts() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        reg.record_check(DependencyCheck::unhealthy("db", true, "a"));
        reg.record_check(DependencyCheck::unhealthy("db", true, "b"));
        let rpt = reg.report();
        let db = rpt.dependencies.iter().find(|d| d.name == "db").unwrap();
        assert_eq!(db.consecutive_failures, 2);
        assert_eq!(db.consecutive_successes, 0);
    }

    #[test]
    fn default_thresholds() {
        let t = DegradedThresholds::default();
        assert_eq!(t.max_latency_us, 5_000_000);
        assert_eq!(t.max_unhealthy_non_critical, 1);
        assert_eq!(t.failure_threshold, 3);
        assert_eq!(t.success_threshold, 1);
    }

    #[test]
    fn readiness_passes_under_threshold() {
        let thresholds = DegradedThresholds {
            failure_threshold: 5,
            ..Default::default()
        };
        let mut reg = HealthRegistry::with_thresholds(thresholds);
        reg.complete_startup();
        reg.record_check(DependencyCheck::unhealthy("db", true, "oops"));
        reg.record_check(DependencyCheck::unhealthy("db", true, "oops"));
        assert_eq!(reg.readiness().status, HealthStatus::Healthy);
    }

    #[test]
    fn multiple_deps_independent() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        reg.record_check(DependencyCheck::healthy("db", true));
        reg.record_check(DependencyCheck::unhealthy("cache", false, "miss"));
        assert_eq!(reg.checks().len(), 2);
        assert_eq!(reg.failure_counts["db"], 0);
        assert_eq!(reg.failure_counts["cache"], 1);
    }

    #[test]
    fn json_null_message_and_latency() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        reg.record_check(DependencyCheck::healthy("db", true));
        let json = reg.report().to_json();
        assert!(json.contains("\"latency_us\":null"));
        assert!(json.contains("\"message\":null"));
    }

    #[test]
    fn aggregate_unknown_critical_dep() {
        let mut reg = HealthRegistry::new();
        reg.complete_startup();
        reg.record_check(DependencyCheck {
            name: "svc".into(),
            status: HealthStatus::Unknown,
            latency_us: None,
            message: None,
            critical: true,
        });
        assert_eq!(reg.aggregate_status(), HealthStatus::Unknown);
    }

    #[test]
    fn liveness_message_content() {
        let mut reg = HealthRegistry::new();
        reg.mark_not_alive();
        let p = reg.liveness();
        assert!(p.message.unwrap().contains("not alive"));
    }

    #[test]
    fn readiness_message_mentions_dep_name() {
        let thresholds = DegradedThresholds {
            failure_threshold: 1,
            ..Default::default()
        };
        let mut reg = HealthRegistry::with_thresholds(thresholds);
        reg.complete_startup();
        reg.record_check(DependencyCheck::unhealthy("postgres", true, "err"));
        let p = reg.readiness();
        assert!(p.message.unwrap().contains("postgres"));
    }
}
