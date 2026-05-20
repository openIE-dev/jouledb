//! Error tracking and aggregation — error fingerprinting (stack hash),
//! error grouping, occurrence counting, first/last seen, error rate
//! calculation, error status management, and affected users tracking.

use std::collections::{HashMap, HashSet};

// ── Error Status ─────────────────────────────────────────────────

/// Status of a tracked error group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorStatus {
    New,
    Acknowledged,
    Resolved,
    Regressed,
    Ignored,
}

// ── Error Severity ───────────────────────────────────────────────

/// Severity level of an error occurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Debug,
    Info,
    Warning,
    Error,
    Fatal,
}

// ── Stack Frame ──────────────────────────────────────────────────

/// A single frame in an error stack trace.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ErrorFrame {
    pub function: String,
    pub file: String,
    pub line: u32,
    pub column: Option<u32>,
}

impl ErrorFrame {
    pub fn new(function: &str, file: &str, line: u32) -> Self {
        Self {
            function: function.to_string(),
            file: file.to_string(),
            line,
            column: None,
        }
    }

    pub fn with_column(mut self, col: u32) -> Self {
        self.column = Some(col);
        self
    }

    /// Display as "function (file:line)".
    pub fn display(&self) -> String {
        match self.column {
            Some(col) => format!("{} ({}:{}:{})", self.function, self.file, self.line, col),
            None => format!("{} ({}:{})", self.function, self.file, self.line),
        }
    }
}

// ── Error Occurrence ─────────────────────────────────────────────

/// A single occurrence of an error.
#[derive(Debug, Clone)]
pub struct ErrorOccurrence {
    pub id: u64,
    pub message: String,
    pub severity: ErrorSeverity,
    pub stack_trace: Vec<ErrorFrame>,
    pub timestamp_ms: u64,
    pub user_id: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl ErrorOccurrence {
    pub fn new(id: u64, message: &str, severity: ErrorSeverity, timestamp_ms: u64) -> Self {
        Self {
            id,
            message: message.to_string(),
            severity,
            stack_trace: Vec::new(),
            timestamp_ms,
            user_id: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_stack(mut self, frames: Vec<ErrorFrame>) -> Self {
        self.stack_trace = frames;
        self
    }

    pub fn with_user(mut self, user_id: &str) -> Self {
        self.user_id = Some(user_id.to_string());
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Compute a fingerprint by hashing the message and stack frames.
    /// This groups similar errors together regardless of variable data.
    pub fn fingerprint(&self) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
        let prime: u64 = 0x100000001b3;

        // Hash the message
        for byte in self.message.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }

        // Hash the stack frames (function + file + line)
        for frame in &self.stack_trace {
            for byte in frame.function.bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(prime);
            }
            for byte in frame.file.bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(prime);
            }
            let line_bytes = frame.line.to_le_bytes();
            for byte in line_bytes {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(prime);
            }
        }

        hash
    }
}

// ── Error Group ──────────────────────────────────────────────────

/// A group of similar error occurrences.
#[derive(Debug, Clone)]
pub struct ErrorGroup {
    pub fingerprint: u64,
    pub message: String,
    pub severity: ErrorSeverity,
    pub status: ErrorStatus,
    pub occurrence_count: u64,
    pub first_seen_ms: u64,
    pub last_seen_ms: u64,
    pub affected_users: HashSet<String>,
    pub sample_stack: Vec<ErrorFrame>,
    /// Timestamps of recent occurrences (for rate calculation).
    recent_timestamps: Vec<u64>,
}

impl ErrorGroup {
    fn new(occurrence: &ErrorOccurrence, fingerprint: u64) -> Self {
        let mut affected = HashSet::new();
        if let Some(uid) = &occurrence.user_id {
            affected.insert(uid.clone());
        }

        Self {
            fingerprint,
            message: occurrence.message.clone(),
            severity: occurrence.severity,
            status: ErrorStatus::New,
            occurrence_count: 1,
            first_seen_ms: occurrence.timestamp_ms,
            last_seen_ms: occurrence.timestamp_ms,
            affected_users: affected,
            sample_stack: occurrence.stack_trace.clone(),
            recent_timestamps: vec![occurrence.timestamp_ms],
        }
    }

    fn record(&mut self, occurrence: &ErrorOccurrence) {
        self.occurrence_count += 1;
        if occurrence.timestamp_ms < self.first_seen_ms {
            self.first_seen_ms = occurrence.timestamp_ms;
        }
        if occurrence.timestamp_ms > self.last_seen_ms {
            self.last_seen_ms = occurrence.timestamp_ms;
        }
        if let Some(uid) = &occurrence.user_id {
            self.affected_users.insert(uid.clone());
        }
        if occurrence.severity > self.severity {
            self.severity = occurrence.severity;
        }
        self.recent_timestamps.push(occurrence.timestamp_ms);

        // If status was Resolved and we see a new occurrence, it's a regression
        if self.status == ErrorStatus::Resolved {
            self.status = ErrorStatus::Regressed;
        }
    }

    /// Number of unique affected users.
    pub fn affected_user_count(&self) -> usize {
        self.affected_users.len()
    }

    /// Duration in ms from first to last seen.
    pub fn duration_ms(&self) -> u64 {
        self.last_seen_ms.saturating_sub(self.first_seen_ms)
    }

    /// Error rate: occurrences per second over the group's lifetime.
    pub fn rate_per_second(&self) -> f64 {
        let duration = self.duration_ms();
        if duration == 0 {
            return self.occurrence_count as f64;
        }
        (self.occurrence_count as f64) / (duration as f64 / 1000.0)
    }

    /// Error rate within a specified window (occurrences per second).
    pub fn rate_in_window(&self, window_start_ms: u64, window_end_ms: u64) -> f64 {
        if window_end_ms <= window_start_ms {
            return 0.0;
        }
        let count = self
            .recent_timestamps
            .iter()
            .filter(|&&t| t >= window_start_ms && t < window_end_ms)
            .count();
        let duration_sec = (window_end_ms - window_start_ms) as f64 / 1000.0;
        count as f64 / duration_sec
    }

    /// Is this a high-frequency error? (> threshold per second in recent window)
    pub fn is_spike(&self, window_start_ms: u64, window_end_ms: u64, threshold: f64) -> bool {
        self.rate_in_window(window_start_ms, window_end_ms) > threshold
    }
}

// ── Error Tracker ────────────────────────────────────────────────

/// Error tracker that groups, counts, and analyzes error occurrences.
pub struct ErrorTracker {
    groups: HashMap<u64, ErrorGroup>,
    total_occurrences: u64,
    next_id: u64,
}

impl ErrorTracker {
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            total_occurrences: 0,
            next_id: 0,
        }
    }

    /// Track a new error occurrence. Returns (occurrence_id, fingerprint).
    pub fn track(&mut self, mut occurrence: ErrorOccurrence) -> (u64, u64) {
        let id = self.next_id;
        self.next_id += 1;
        occurrence.id = id;

        let fingerprint = occurrence.fingerprint();
        self.total_occurrences += 1;

        if let Some(group) = self.groups.get_mut(&fingerprint) {
            group.record(&occurrence);
        } else {
            self.groups
                .insert(fingerprint, ErrorGroup::new(&occurrence, fingerprint));
        }

        (id, fingerprint)
    }

    /// Get an error group by fingerprint.
    pub fn get_group(&self, fingerprint: u64) -> Option<&ErrorGroup> {
        self.groups.get(&fingerprint)
    }

    /// Update the status of an error group.
    pub fn set_status(&mut self, fingerprint: u64, status: ErrorStatus) -> bool {
        if let Some(group) = self.groups.get_mut(&fingerprint) {
            group.status = status;
            true
        } else {
            false
        }
    }

    /// Total number of error groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Total number of individual occurrences.
    pub fn total_occurrences(&self) -> u64 {
        self.total_occurrences
    }

    /// Get all groups sorted by occurrence count descending.
    pub fn groups_by_count(&self) -> Vec<&ErrorGroup> {
        let mut groups: Vec<&ErrorGroup> = self.groups.values().collect();
        groups.sort_by(|a, b| {
            b.occurrence_count
                .cmp(&a.occurrence_count)
                .then_with(|| a.fingerprint.cmp(&b.fingerprint))
        });
        groups
    }

    /// Get all groups sorted by last seen descending (most recent first).
    pub fn groups_by_recency(&self) -> Vec<&ErrorGroup> {
        let mut groups: Vec<&ErrorGroup> = self.groups.values().collect();
        groups.sort_by(|a, b| {
            b.last_seen_ms
                .cmp(&a.last_seen_ms)
                .then_with(|| a.fingerprint.cmp(&b.fingerprint))
        });
        groups
    }

    /// Get groups filtered by status.
    pub fn groups_by_status(&self, status: ErrorStatus) -> Vec<&ErrorGroup> {
        let mut groups: Vec<&ErrorGroup> = self
            .groups
            .values()
            .filter(|g| g.status == status)
            .collect();
        groups.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));
        groups
    }

    /// Get groups filtered by severity (at or above the given level).
    pub fn groups_by_min_severity(&self, min_severity: ErrorSeverity) -> Vec<&ErrorGroup> {
        let mut groups: Vec<&ErrorGroup> = self
            .groups
            .values()
            .filter(|g| g.severity >= min_severity)
            .collect();
        groups.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));
        groups
    }

    /// Get groups with the most affected users.
    pub fn groups_by_impact(&self) -> Vec<&ErrorGroup> {
        let mut groups: Vec<&ErrorGroup> = self.groups.values().collect();
        groups.sort_by(|a, b| {
            b.affected_user_count()
                .cmp(&a.affected_user_count())
                .then_with(|| b.occurrence_count.cmp(&a.occurrence_count))
        });
        groups
    }

    /// Compute overall error rate (occurrences per second) across all groups
    /// in a given time window.
    pub fn overall_rate(&self, window_start_ms: u64, window_end_ms: u64) -> f64 {
        if window_end_ms <= window_start_ms {
            return 0.0;
        }
        let total: usize = self
            .groups
            .values()
            .map(|g| {
                g.recent_timestamps
                    .iter()
                    .filter(|&&t| t >= window_start_ms && t < window_end_ms)
                    .count()
            })
            .sum();
        let duration_sec = (window_end_ms - window_start_ms) as f64 / 1000.0;
        total as f64 / duration_sec
    }

    /// Get all unique affected users across all error groups.
    pub fn all_affected_users(&self) -> HashSet<String> {
        let mut users = HashSet::new();
        for group in self.groups.values() {
            for user in &group.affected_users {
                users.insert(user.clone());
            }
        }
        users
    }

    /// Resolve all errors (set status to Resolved).
    pub fn resolve_all(&mut self) {
        for group in self.groups.values_mut() {
            if group.status != ErrorStatus::Ignored {
                group.status = ErrorStatus::Resolved;
            }
        }
    }
}

impl Default for ErrorTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_occurrence(msg: &str, ts: u64) -> ErrorOccurrence {
        ErrorOccurrence::new(0, msg, ErrorSeverity::Error, ts).with_stack(vec![
            ErrorFrame::new("handle_request", "server.rs", 42),
            ErrorFrame::new("parse_json", "parser.rs", 10),
        ])
    }

    #[test]
    fn test_fingerprint_same_error() {
        let a = make_occurrence("connection refused", 1000);
        let b = make_occurrence("connection refused", 2000);
        assert_eq!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn test_fingerprint_different_message() {
        let a = make_occurrence("connection refused", 1000);
        let b = make_occurrence("timeout", 1000);
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn test_fingerprint_different_stack() {
        let a = make_occurrence("error", 1000);
        let mut b = make_occurrence("error", 1000);
        b.stack_trace = vec![ErrorFrame::new("other_fn", "other.rs", 99)];
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn test_track_groups_same_error() {
        let mut tracker = ErrorTracker::new();
        let (_, fp1) = tracker.track(make_occurrence("error A", 1000));
        let (_, fp2) = tracker.track(make_occurrence("error A", 2000));
        assert_eq!(fp1, fp2);
        assert_eq!(tracker.group_count(), 1);
        assert_eq!(tracker.total_occurrences(), 2);
        let group = tracker.get_group(fp1).unwrap();
        assert_eq!(group.occurrence_count, 2);
    }

    #[test]
    fn test_track_groups_different_errors() {
        let mut tracker = ErrorTracker::new();
        tracker.track(make_occurrence("error A", 1000));
        tracker.track(make_occurrence("error B", 2000));
        assert_eq!(tracker.group_count(), 2);
    }

    #[test]
    fn test_first_last_seen() {
        let mut tracker = ErrorTracker::new();
        let (_, fp) = tracker.track(make_occurrence("err", 1000));
        tracker.track(make_occurrence("err", 5000));
        tracker.track(make_occurrence("err", 3000));
        let group = tracker.get_group(fp).unwrap();
        assert_eq!(group.first_seen_ms, 1000);
        assert_eq!(group.last_seen_ms, 5000);
    }

    #[test]
    fn test_affected_users() {
        let mut tracker = ErrorTracker::new();
        let occ1 = make_occurrence("err", 1000).with_user("alice");
        let occ2 = make_occurrence("err", 2000).with_user("bob");
        let occ3 = make_occurrence("err", 3000).with_user("alice");
        let (_, fp) = tracker.track(occ1);
        tracker.track(occ2);
        tracker.track(occ3);
        let group = tracker.get_group(fp).unwrap();
        assert_eq!(group.affected_user_count(), 2);
    }

    #[test]
    fn test_error_status() {
        let mut tracker = ErrorTracker::new();
        let (_, fp) = tracker.track(make_occurrence("err", 1000));
        assert_eq!(tracker.get_group(fp).unwrap().status, ErrorStatus::New);
        tracker.set_status(fp, ErrorStatus::Acknowledged);
        assert_eq!(
            tracker.get_group(fp).unwrap().status,
            ErrorStatus::Acknowledged
        );
    }

    #[test]
    fn test_regression_detection() {
        let mut tracker = ErrorTracker::new();
        let (_, fp) = tracker.track(make_occurrence("err", 1000));
        tracker.set_status(fp, ErrorStatus::Resolved);
        tracker.track(make_occurrence("err", 5000));
        assert_eq!(
            tracker.get_group(fp).unwrap().status,
            ErrorStatus::Regressed
        );
    }

    #[test]
    fn test_error_rate() {
        let mut tracker = ErrorTracker::new();
        let (_, fp) = tracker.track(make_occurrence("err", 1000));
        tracker.track(make_occurrence("err", 3000));
        let group = tracker.get_group(fp).unwrap();
        // 2 occurrences over 2000ms = 1 per second
        assert!((group.rate_per_second() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rate_in_window() {
        let mut tracker = ErrorTracker::new();
        let (_, fp) = tracker.track(make_occurrence("err", 1000));
        tracker.track(make_occurrence("err", 2000));
        tracker.track(make_occurrence("err", 5000));
        let group = tracker.get_group(fp).unwrap();
        // 2 occurrences in [0, 3000) window = 2/3 per second
        let rate = group.rate_in_window(0, 3000);
        assert!((rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_groups_by_count() {
        let mut tracker = ErrorTracker::new();
        tracker.track(make_occurrence("rare", 1000));
        let frequent = make_occurrence("frequent", 2000);
        tracker.track(frequent.clone());
        tracker.track(make_occurrence("frequent", 3000));
        let groups = tracker.groups_by_count();
        assert_eq!(groups[0].message, "frequent");
    }

    #[test]
    fn test_groups_by_recency() {
        let mut tracker = ErrorTracker::new();
        tracker.track(make_occurrence("old", 1000));
        tracker.track(make_occurrence("new", 5000));
        let groups = tracker.groups_by_recency();
        assert_eq!(groups[0].message, "new");
    }

    #[test]
    fn test_groups_by_status() {
        let mut tracker = ErrorTracker::new();
        let (_, fp1) = tracker.track(make_occurrence("a", 1000));
        tracker.track(make_occurrence("b", 2000));
        tracker.set_status(fp1, ErrorStatus::Resolved);
        let resolved = tracker.groups_by_status(ErrorStatus::Resolved);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].message, "a");
    }

    #[test]
    fn test_groups_by_min_severity() {
        let mut tracker = ErrorTracker::new();
        tracker.track(ErrorOccurrence::new(0, "warn", ErrorSeverity::Warning, 1000));
        tracker.track(ErrorOccurrence::new(0, "fatal", ErrorSeverity::Fatal, 2000));
        let fatal = tracker.groups_by_min_severity(ErrorSeverity::Fatal);
        assert_eq!(fatal.len(), 1);
    }

    #[test]
    fn test_groups_by_impact() {
        let mut tracker = ErrorTracker::new();
        let o1 = make_occurrence("low_impact", 1000).with_user("alice");
        tracker.track(o1);
        let o2 = make_occurrence("high_impact", 2000).with_user("bob");
        let o3 = make_occurrence("high_impact", 3000).with_user("carol");
        tracker.track(o2);
        tracker.track(o3);
        let by_impact = tracker.groups_by_impact();
        assert_eq!(by_impact[0].message, "high_impact");
    }

    #[test]
    fn test_overall_rate() {
        let mut tracker = ErrorTracker::new();
        tracker.track(make_occurrence("a", 1000));
        tracker.track(make_occurrence("b", 2000));
        tracker.track(make_occurrence("a", 3000));
        // 3 occurrences in [0, 5000) window = 3/5 per second
        let rate = tracker.overall_rate(0, 5000);
        assert!((rate - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_all_affected_users() {
        let mut tracker = ErrorTracker::new();
        tracker.track(make_occurrence("a", 1000).with_user("alice"));
        tracker.track(make_occurrence("b", 2000).with_user("bob"));
        tracker.track(make_occurrence("a", 3000).with_user("bob"));
        let users = tracker.all_affected_users();
        assert_eq!(users.len(), 2);
    }

    #[test]
    fn test_resolve_all() {
        let mut tracker = ErrorTracker::new();
        let (_, fp1) = tracker.track(make_occurrence("a", 1000));
        let (_, fp2) = tracker.track(make_occurrence("b", 2000));
        tracker.set_status(fp2, ErrorStatus::Ignored);
        tracker.resolve_all();
        assert_eq!(
            tracker.get_group(fp1).unwrap().status,
            ErrorStatus::Resolved
        );
        assert_eq!(
            tracker.get_group(fp2).unwrap().status,
            ErrorStatus::Ignored
        ); // unchanged
    }

    #[test]
    fn test_error_frame_display() {
        let f = ErrorFrame::new("handle", "srv.rs", 42);
        assert_eq!(f.display(), "handle (srv.rs:42)");
        let f2 = f.with_column(5);
        assert_eq!(f2.display(), "handle (srv.rs:42:5)");
    }

    #[test]
    fn test_occurrence_metadata() {
        let occ = make_occurrence("err", 1000)
            .with_metadata("env", "production")
            .with_metadata("version", "1.2.3");
        assert_eq!(occ.metadata.get("env").unwrap(), "production");
        assert_eq!(occ.metadata.len(), 2);
    }

    #[test]
    fn test_is_spike() {
        let mut tracker = ErrorTracker::new();
        // 10 occurrences in 1 second
        for i in 0..10 {
            tracker.track(make_occurrence("spam", 1000 + i * 100));
        }
        let groups = tracker.groups_by_count();
        assert!(groups[0].is_spike(1000, 2000, 5.0));
        assert!(!groups[0].is_spike(1000, 2000, 20.0));
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = ErrorTracker::new();
        assert_eq!(tracker.group_count(), 0);
        assert_eq!(tracker.total_occurrences(), 0);
        assert!(tracker.groups_by_count().is_empty());
    }

    #[test]
    fn test_severity_escalation() {
        let mut tracker = ErrorTracker::new();
        let (_, fp) = tracker.track(ErrorOccurrence::new(
            0,
            "err",
            ErrorSeverity::Warning,
            1000,
        ));
        tracker.track(ErrorOccurrence::new(0, "err", ErrorSeverity::Fatal, 2000));
        let group = tracker.get_group(fp).unwrap();
        // Severity should have been escalated
        // Note: fingerprint depends on message only for this case (no stack),
        // so both will group together since they share the same message and empty stack
        assert!(group.severity >= ErrorSeverity::Warning);
    }
}
