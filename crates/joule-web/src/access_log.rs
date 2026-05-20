//! Access logging — structured access events with who/what/when/where/outcome fields,
//! log rotation, query by time/user/resource, compliance reporting, access pattern
//! analysis, and suspicious access detection.
//!
//! Replaces morgan, winston-access, and access-log middleware with a pure-Rust
//! structured access logger supporting compliance queries and anomaly detection.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Access log errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessLogError {
    /// Log segment is sealed/read-only.
    SegmentSealed(String),
    /// Entry not found.
    EntryNotFound(u64),
    /// Invalid time range.
    InvalidTimeRange { start: u64, end: u64 },
}

impl fmt::Display for AccessLogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SegmentSealed(name) => write!(f, "log segment sealed: {name}"),
            Self::EntryNotFound(id) => write!(f, "entry not found: {id}"),
            Self::InvalidTimeRange { start, end } => {
                write!(f, "invalid time range: {start}..{end}")
            }
        }
    }
}

impl std::error::Error for AccessLogError {}

// ── Types ──────────────────────────────────────────────────────

/// Outcome of an access attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessOutcome {
    Allowed,
    Denied,
    Error,
    Timeout,
    RateLimited,
}

impl AccessOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Allowed => "allowed",
            Self::Denied => "denied",
            Self::Error => "error",
            Self::Timeout => "timeout",
            Self::RateLimited => "rate_limited",
        }
    }

    pub fn is_successful(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

impl fmt::Display for AccessOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Severity level for suspicious activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl AlertSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }
}

/// A structured access event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessEvent {
    /// Monotonic sequence ID.
    pub id: u64,
    /// Timestamp in epoch milliseconds.
    pub timestamp_ms: u64,

    // Who
    /// User/actor ID.
    pub user_id: String,
    /// Session ID (if applicable).
    pub session_id: Option<String>,
    /// User role/group.
    pub role: Option<String>,

    // What
    /// The action performed (e.g., "read", "write", "delete").
    pub action: String,
    /// The resource accessed (e.g., "document:42", "/api/users").
    pub resource: String,
    /// Resource type category.
    pub resource_type: Option<String>,

    // Where
    /// Source IP address.
    pub source_ip: Option<String>,
    /// User agent string.
    pub user_agent: Option<String>,
    /// Geographic region (if available).
    pub region: Option<String>,

    // Outcome
    /// Whether access was allowed or denied.
    pub outcome: AccessOutcome,
    /// HTTP status code or application result code.
    pub status_code: Option<u16>,
    /// Duration of the access in microseconds.
    pub duration_us: Option<u64>,

    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl AccessEvent {
    pub fn new(user_id: &str, action: &str, resource: &str, outcome: AccessOutcome) -> Self {
        Self {
            id: 0,
            timestamp_ms: 0,
            user_id: user_id.to_string(),
            session_id: None,
            role: None,
            action: action.to_string(),
            resource: resource.to_string(),
            resource_type: None,
            source_ip: None,
            user_agent: None,
            region: None,
            outcome,
            status_code: None,
            duration_us: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_timestamp(mut self, ms: u64) -> Self {
        self.timestamp_ms = ms;
        self
    }

    pub fn with_session(mut self, session_id: &str) -> Self {
        self.session_id = Some(session_id.to_string());
        self
    }

    pub fn with_role(mut self, role: &str) -> Self {
        self.role = Some(role.to_string());
        self
    }

    pub fn with_source_ip(mut self, ip: &str) -> Self {
        self.source_ip = Some(ip.to_string());
        self
    }

    pub fn with_user_agent(mut self, ua: &str) -> Self {
        self.user_agent = Some(ua.to_string());
        self
    }

    pub fn with_region(mut self, region: &str) -> Self {
        self.region = Some(region.to_string());
        self
    }

    pub fn with_status_code(mut self, code: u16) -> Self {
        self.status_code = Some(code);
        self
    }

    pub fn with_duration_us(mut self, us: u64) -> Self {
        self.duration_us = Some(us);
        self
    }

    pub fn with_resource_type(mut self, rt: &str) -> Self {
        self.resource_type = Some(rt.to_string());
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// A log segment (for rotation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogSegment {
    /// Segment name/identifier.
    pub name: String,
    /// Events in this segment.
    pub events: Vec<AccessEvent>,
    /// Whether this segment is sealed (no more writes).
    pub sealed: bool,
    /// Start time (first event timestamp) in epoch millis.
    pub start_ms: u64,
    /// End time (last event timestamp) in epoch millis.
    pub end_ms: u64,
}

impl LogSegment {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            events: Vec::new(),
            sealed: false,
            start_ms: 0,
            end_ms: 0,
        }
    }

    /// Seal this segment (no further writes).
    pub fn seal(&mut self) {
        self.sealed = true;
    }

    /// Event count.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the segment is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

/// Query filter for searching access events.
#[derive(Debug, Clone, Default)]
pub struct AccessQuery {
    /// Filter by user ID.
    pub user_id: Option<String>,
    /// Filter by action.
    pub action: Option<String>,
    /// Filter by resource (prefix match).
    pub resource_prefix: Option<String>,
    /// Filter by outcome.
    pub outcome: Option<AccessOutcome>,
    /// Filter by time range (start inclusive).
    pub start_ms: Option<u64>,
    /// Filter by time range (end inclusive).
    pub end_ms: Option<u64>,
    /// Filter by source IP.
    pub source_ip: Option<String>,
    /// Filter by role.
    pub role: Option<String>,
    /// Maximum results.
    pub limit: Option<usize>,
}

impl AccessQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_user(mut self, user_id: &str) -> Self {
        self.user_id = Some(user_id.to_string());
        self
    }

    pub fn with_action(mut self, action: &str) -> Self {
        self.action = Some(action.to_string());
        self
    }

    pub fn with_resource_prefix(mut self, prefix: &str) -> Self {
        self.resource_prefix = Some(prefix.to_string());
        self
    }

    pub fn with_outcome(mut self, outcome: AccessOutcome) -> Self {
        self.outcome = Some(outcome);
        self
    }

    pub fn with_time_range(mut self, start_ms: u64, end_ms: u64) -> Self {
        self.start_ms = Some(start_ms);
        self.end_ms = Some(end_ms);
        self
    }

    pub fn with_source_ip(mut self, ip: &str) -> Self {
        self.source_ip = Some(ip.to_string());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Check if an event matches this query.
    fn matches(&self, event: &AccessEvent) -> bool {
        if let Some(uid) = &self.user_id {
            if event.user_id != *uid {
                return false;
            }
        }
        if let Some(action) = &self.action {
            if event.action != *action {
                return false;
            }
        }
        if let Some(prefix) = &self.resource_prefix {
            if !event.resource.starts_with(prefix.as_str()) {
                return false;
            }
        }
        if let Some(outcome) = &self.outcome {
            if event.outcome != *outcome {
                return false;
            }
        }
        if let Some(start) = self.start_ms {
            if event.timestamp_ms < start {
                return false;
            }
        }
        if let Some(end) = self.end_ms {
            if event.timestamp_ms > end {
                return false;
            }
        }
        if let Some(ip) = &self.source_ip {
            if event.source_ip.as_deref() != Some(ip.as_str()) {
                return false;
            }
        }
        if let Some(role) = &self.role {
            if event.role.as_deref() != Some(role.as_str()) {
                return false;
            }
        }
        true
    }
}

/// Suspicious access alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousAlert {
    /// Description of the suspicious activity.
    pub description: String,
    /// Severity level.
    pub severity: AlertSeverity,
    /// User involved.
    pub user_id: String,
    /// Number of events that triggered the alert.
    pub event_count: usize,
    /// Time window in milliseconds.
    pub window_ms: u64,
    /// Timestamp of the alert.
    pub timestamp_ms: u64,
}

/// Compliance report summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Total events in the period.
    pub total_events: usize,
    /// Events by outcome.
    pub by_outcome: HashMap<String, usize>,
    /// Events by action.
    pub by_action: HashMap<String, usize>,
    /// Unique users.
    pub unique_users: usize,
    /// Denial count.
    pub denial_count: usize,
    /// Error count.
    pub error_count: usize,
    /// Time range start.
    pub start_ms: u64,
    /// Time range end.
    pub end_ms: u64,
}

/// The access logger.
pub struct AccessLogger {
    /// Current active segment.
    current_segment: LogSegment,
    /// Archived (sealed) segments.
    archived_segments: Vec<LogSegment>,
    /// Next event ID.
    next_id: u64,
    /// Maximum events per segment before rotation.
    pub max_events_per_segment: usize,
    /// Maximum number of archived segments to keep.
    pub max_archived_segments: usize,
    /// Thresholds for suspicious access detection.
    pub denial_threshold: usize,
    /// Time window for denial threshold (ms).
    pub denial_window_ms: u64,
    /// Alerts generated.
    alerts: Vec<SuspiciousAlert>,
}

impl AccessLogger {
    pub fn new() -> Self {
        Self {
            current_segment: LogSegment::new("segment-0"),
            archived_segments: Vec::new(),
            next_id: 1,
            max_events_per_segment: 10_000,
            max_archived_segments: 10,
            denial_threshold: 10,
            denial_window_ms: 60_000,
            alerts: Vec::new(),
        }
    }

    /// Log an access event. Handles ID assignment and rotation.
    pub fn log(&mut self, mut event: AccessEvent) -> u64 {
        event.id = self.next_id;
        self.next_id += 1;

        let ts = event.timestamp_ms;

        // Update segment time bounds.
        if self.current_segment.is_empty() {
            self.current_segment.start_ms = ts;
        }
        self.current_segment.end_ms = ts;

        // Check for suspicious patterns before adding.
        self.check_suspicious(&event);

        self.current_segment.events.push(event);

        // Rotate if needed.
        if self.current_segment.len() >= self.max_events_per_segment {
            self.rotate();
        }

        self.next_id - 1
    }

    /// Force rotation of the current segment.
    pub fn rotate(&mut self) {
        let mut old = std::mem::replace(
            &mut self.current_segment,
            LogSegment::new(&format!("segment-{}", self.archived_segments.len() + 1)),
        );
        old.seal();
        self.archived_segments.push(old);

        // Trim old segments.
        while self.archived_segments.len() > self.max_archived_segments {
            self.archived_segments.remove(0);
        }
    }

    /// Query events across all segments.
    pub fn query(&self, q: &AccessQuery) -> Vec<&AccessEvent> {
        let mut results: Vec<&AccessEvent> = Vec::new();
        let limit = q.limit.unwrap_or(usize::MAX);

        // Search archived segments (oldest first).
        for segment in &self.archived_segments {
            for event in &segment.events {
                if results.len() >= limit {
                    return results;
                }
                if q.matches(event) {
                    results.push(event);
                }
            }
        }

        // Search current segment.
        for event in &self.current_segment.events {
            if results.len() >= limit {
                return results;
            }
            if q.matches(event) {
                results.push(event);
            }
        }

        results
    }

    /// Total event count across all segments.
    pub fn total_events(&self) -> usize {
        self.archived_segments.iter().map(|s| s.len()).sum::<usize>()
            + self.current_segment.len()
    }

    /// Number of segments (including current).
    pub fn segment_count(&self) -> usize {
        self.archived_segments.len() + 1
    }

    /// Get all events in the current segment.
    pub fn current_events(&self) -> &[AccessEvent] {
        &self.current_segment.events
    }

    /// Generate a compliance report for a time range.
    pub fn compliance_report(&self, start_ms: u64, end_ms: u64) -> ComplianceReport {
        let mut by_outcome: HashMap<String, usize> = HashMap::new();
        let mut by_action: HashMap<String, usize> = HashMap::new();
        let mut users = std::collections::HashSet::new();
        let mut denial_count = 0;
        let mut error_count = 0;
        let mut total = 0;

        let query = AccessQuery::new().with_time_range(start_ms, end_ms);
        let events = self.query(&query);

        for event in &events {
            total += 1;
            *by_outcome.entry(event.outcome.as_str().to_string()).or_insert(0) += 1;
            *by_action.entry(event.action.clone()).or_insert(0) += 1;
            users.insert(event.user_id.clone());
            if event.outcome == AccessOutcome::Denied {
                denial_count += 1;
            }
            if event.outcome == AccessOutcome::Error {
                error_count += 1;
            }
        }

        ComplianceReport {
            total_events: total,
            by_outcome,
            by_action,
            unique_users: users.len(),
            denial_count,
            error_count,
            start_ms,
            end_ms,
        }
    }

    /// Analyze access patterns for a user: returns (action, count) sorted by frequency.
    pub fn user_access_pattern(&self, user_id: &str) -> Vec<(String, usize)> {
        let query = AccessQuery::new().with_user(user_id);
        let events = self.query(&query);
        let mut counts: HashMap<String, usize> = HashMap::new();
        for event in events {
            *counts.entry(event.action.clone()).or_insert(0) += 1;
        }
        let mut result: Vec<(String, usize)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    /// Check for suspicious access and generate alerts.
    fn check_suspicious(&mut self, event: &AccessEvent) {
        if event.outcome != AccessOutcome::Denied {
            return;
        }
        // Count recent denials for this user.
        let window_start = event.timestamp_ms.saturating_sub(self.denial_window_ms);
        let user_id = &event.user_id;
        let recent_denials = self
            .current_segment
            .events
            .iter()
            .filter(|e| {
                e.user_id == *user_id
                    && e.outcome == AccessOutcome::Denied
                    && e.timestamp_ms >= window_start
            })
            .count();

        if recent_denials + 1 >= self.denial_threshold {
            self.alerts.push(SuspiciousAlert {
                description: format!(
                    "User '{user_id}' had {} access denials in {}ms window",
                    recent_denials + 1,
                    self.denial_window_ms
                ),
                severity: if recent_denials + 1 >= self.denial_threshold * 2 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::High
                },
                user_id: user_id.clone(),
                event_count: recent_denials + 1,
                window_ms: self.denial_window_ms,
                timestamp_ms: event.timestamp_ms,
            });
        }
    }

    /// Get generated alerts.
    pub fn alerts(&self) -> &[SuspiciousAlert] {
        &self.alerts
    }

    /// Clear alerts.
    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
    }

    /// Export all events as JSON.
    pub fn export_json(&self) -> Vec<serde_json::Value> {
        let all_events: Vec<&AccessEvent> = self
            .archived_segments
            .iter()
            .flat_map(|s| s.events.iter())
            .chain(self.current_segment.events.iter())
            .collect();

        all_events
            .iter()
            .filter_map(|e| serde_json::to_value(e).ok())
            .collect()
    }
}

impl Default for AccessLogger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(user: &str, action: &str, resource: &str, outcome: AccessOutcome, ts: u64) -> AccessEvent {
        AccessEvent::new(user, action, resource, outcome).with_timestamp(ts)
    }

    #[test]
    fn test_log_event() {
        let mut logger = AccessLogger::new();
        let id = logger.log(make_event("alice", "read", "/docs/1", AccessOutcome::Allowed, 1000));
        assert_eq!(id, 1);
        assert_eq!(logger.total_events(), 1);
    }

    #[test]
    fn test_log_assigns_sequential_ids() {
        let mut logger = AccessLogger::new();
        let id1 = logger.log(make_event("a", "read", "/x", AccessOutcome::Allowed, 1000));
        let id2 = logger.log(make_event("b", "write", "/y", AccessOutcome::Allowed, 2000));
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_query_by_user() {
        let mut logger = AccessLogger::new();
        logger.log(make_event("alice", "read", "/x", AccessOutcome::Allowed, 1000));
        logger.log(make_event("bob", "read", "/x", AccessOutcome::Allowed, 2000));
        logger.log(make_event("alice", "write", "/y", AccessOutcome::Allowed, 3000));

        let results = logger.query(&AccessQuery::new().with_user("alice"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_action() {
        let mut logger = AccessLogger::new();
        logger.log(make_event("a", "read", "/x", AccessOutcome::Allowed, 1000));
        logger.log(make_event("a", "write", "/x", AccessOutcome::Allowed, 2000));
        logger.log(make_event("a", "read", "/y", AccessOutcome::Allowed, 3000));

        let results = logger.query(&AccessQuery::new().with_action("read"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_outcome() {
        let mut logger = AccessLogger::new();
        logger.log(make_event("a", "read", "/x", AccessOutcome::Allowed, 1000));
        logger.log(make_event("a", "read", "/y", AccessOutcome::Denied, 2000));

        let results = logger.query(&AccessQuery::new().with_outcome(AccessOutcome::Denied));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].resource, "/y");
    }

    #[test]
    fn test_query_by_time_range() {
        let mut logger = AccessLogger::new();
        logger.log(make_event("a", "read", "/x", AccessOutcome::Allowed, 1000));
        logger.log(make_event("a", "read", "/y", AccessOutcome::Allowed, 2000));
        logger.log(make_event("a", "read", "/z", AccessOutcome::Allowed, 3000));

        let results = logger.query(&AccessQuery::new().with_time_range(1500, 2500));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].resource, "/y");
    }

    #[test]
    fn test_query_by_resource_prefix() {
        let mut logger = AccessLogger::new();
        logger.log(make_event("a", "read", "/api/users/1", AccessOutcome::Allowed, 1000));
        logger.log(make_event("a", "read", "/api/docs/1", AccessOutcome::Allowed, 2000));
        logger.log(make_event("a", "read", "/api/users/2", AccessOutcome::Allowed, 3000));

        let results = logger.query(&AccessQuery::new().with_resource_prefix("/api/users"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_with_limit() {
        let mut logger = AccessLogger::new();
        for i in 0..10 {
            logger.log(make_event("a", "read", "/x", AccessOutcome::Allowed, i * 1000));
        }
        let results = logger.query(&AccessQuery::new().with_limit(3));
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_rotation() {
        let mut logger = AccessLogger::new();
        logger.max_events_per_segment = 3;
        for i in 0..7 {
            logger.log(make_event("a", "read", "/x", AccessOutcome::Allowed, i * 1000));
        }
        assert_eq!(logger.segment_count(), 3); // 2 archived + 1 current
        assert_eq!(logger.total_events(), 7);
    }

    #[test]
    fn test_rotation_trims_old_segments() {
        let mut logger = AccessLogger::new();
        logger.max_events_per_segment = 2;
        logger.max_archived_segments = 2;
        for i in 0..10 {
            logger.log(make_event("a", "read", "/x", AccessOutcome::Allowed, i * 1000));
        }
        // Should keep only 2 archived + 1 current = 3 segments
        assert!(logger.segment_count() <= 3);
    }

    #[test]
    fn test_compliance_report() {
        let mut logger = AccessLogger::new();
        logger.log(make_event("alice", "read", "/x", AccessOutcome::Allowed, 1000));
        logger.log(make_event("bob", "write", "/y", AccessOutcome::Denied, 2000));
        logger.log(make_event("alice", "read", "/z", AccessOutcome::Error, 3000));

        let report = logger.compliance_report(0, 5000);
        assert_eq!(report.total_events, 3);
        assert_eq!(report.unique_users, 2);
        assert_eq!(report.denial_count, 1);
        assert_eq!(report.error_count, 1);
    }

    #[test]
    fn test_user_access_pattern() {
        let mut logger = AccessLogger::new();
        logger.log(make_event("alice", "read", "/x", AccessOutcome::Allowed, 1000));
        logger.log(make_event("alice", "read", "/y", AccessOutcome::Allowed, 2000));
        logger.log(make_event("alice", "write", "/z", AccessOutcome::Allowed, 3000));

        let patterns = logger.user_access_pattern("alice");
        assert_eq!(patterns.len(), 2);
        // read should be first (count=2)
        assert_eq!(patterns[0].0, "read");
        assert_eq!(patterns[0].1, 2);
    }

    #[test]
    fn test_suspicious_access_detection() {
        let mut logger = AccessLogger::new();
        logger.denial_threshold = 3;
        logger.denial_window_ms = 10_000;

        // Log 3 denials rapidly
        for i in 0..3 {
            logger.log(make_event(
                "attacker",
                "read",
                "/admin",
                AccessOutcome::Denied,
                1000 + i * 100,
            ));
        }

        assert!(!logger.alerts().is_empty());
        assert_eq!(logger.alerts()[0].user_id, "attacker");
    }

    #[test]
    fn test_no_alert_below_threshold() {
        let mut logger = AccessLogger::new();
        logger.denial_threshold = 5;
        logger.log(make_event("user", "read", "/x", AccessOutcome::Denied, 1000));
        logger.log(make_event("user", "read", "/x", AccessOutcome::Denied, 2000));
        assert!(logger.alerts().is_empty());
    }

    #[test]
    fn test_event_builder() {
        let event = AccessEvent::new("alice", "read", "/docs", AccessOutcome::Allowed)
            .with_timestamp(5000)
            .with_session("sess-123")
            .with_role("admin")
            .with_source_ip("10.0.0.1")
            .with_user_agent("Mozilla/5.0")
            .with_status_code(200)
            .with_duration_us(1500)
            .with_resource_type("document")
            .with_metadata("tenant", "acme");

        assert_eq!(event.session_id, Some("sess-123".into()));
        assert_eq!(event.source_ip, Some("10.0.0.1".into()));
        assert_eq!(event.status_code, Some(200));
        assert_eq!(event.duration_us, Some(1500));
    }

    #[test]
    fn test_export_json() {
        let mut logger = AccessLogger::new();
        logger.log(make_event("a", "read", "/x", AccessOutcome::Allowed, 1000));
        let exported = logger.export_json();
        assert_eq!(exported.len(), 1);
        assert!(exported[0].get("user_id").is_some());
    }

    #[test]
    fn test_clear_alerts() {
        let mut logger = AccessLogger::new();
        logger.denial_threshold = 1;
        logger.log(make_event("u", "r", "/x", AccessOutcome::Denied, 1000));
        assert!(!logger.alerts().is_empty());
        logger.clear_alerts();
        assert!(logger.alerts().is_empty());
    }

    #[test]
    fn test_query_by_source_ip() {
        let mut logger = AccessLogger::new();
        logger.log(
            make_event("a", "read", "/x", AccessOutcome::Allowed, 1000)
                .with_source_ip("10.0.0.1"),
        );
        logger.log(
            make_event("a", "read", "/y", AccessOutcome::Allowed, 2000)
                .with_source_ip("10.0.0.2"),
        );
        let results = logger.query(&AccessQuery::new().with_source_ip("10.0.0.1"));
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_outcome_display() {
        assert_eq!(AccessOutcome::Allowed.to_string(), "allowed");
        assert_eq!(AccessOutcome::Denied.to_string(), "denied");
        assert!(AccessOutcome::Allowed.is_successful());
        assert!(!AccessOutcome::Denied.is_successful());
    }

    #[test]
    fn test_query_across_segments() {
        let mut logger = AccessLogger::new();
        logger.max_events_per_segment = 2;
        logger.log(make_event("a", "read", "/x", AccessOutcome::Allowed, 1000));
        logger.log(make_event("a", "write", "/y", AccessOutcome::Allowed, 2000));
        // Trigger rotation
        logger.log(make_event("a", "read", "/z", AccessOutcome::Allowed, 3000));

        let all = logger.query(&AccessQuery::new().with_user("a"));
        assert_eq!(all.len(), 3);
    }
}
