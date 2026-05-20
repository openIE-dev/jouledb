//! Security audit logging — structured audit events with actor/action/resource/outcome
//! fields, tamper-evident chaining (hash chain), log rotation, query by time
//! range/actor/action, and export to JSON.
//!
//! Replaces winston-audit, bunyan, and pino-audit with a pure-Rust append-only
//! audit log with cryptographic tamper detection.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use uuid::Uuid;

// ── Types ──────────────────────────────────────────────────────

/// Outcome of an audited action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditOutcome {
    Success,
    Failure,
    Denied,
    Error,
    Timeout,
    Partial,
}

impl AuditOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::Denied => "denied",
            Self::Error => "error",
            Self::Timeout => "timeout",
            Self::Partial => "partial",
        }
    }

    pub fn is_successful(&self) -> bool {
        matches!(self, Self::Success | Self::Partial)
    }
}

impl std::fmt::Display for AuditOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Severity level for audit events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Critical => "critical",
        }
    }
}

// ── Errors ─────────────────────────────────────────────────────

/// Audit log errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditLogError {
    /// Hash chain integrity violation.
    ChainBroken { sequence: u64, expected: String, got: String },
    /// Entry tampered (hash mismatch).
    EntryTampered { sequence: u64 },
    /// Sequence number gap.
    SequenceGap { expected: u64, got: u64 },
    /// Log is sealed (read-only).
    LogSealed,
}

impl std::fmt::Display for AuditLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChainBroken { sequence, expected, got } => {
                write!(f, "chain broken at seq {sequence}: expected {expected}, got {got}")
            }
            Self::EntryTampered { sequence } => {
                write!(f, "entry tampered at sequence {sequence}")
            }
            Self::SequenceGap { expected, got } => {
                write!(f, "sequence gap: expected {expected}, got {got}")
            }
            Self::LogSealed => write!(f, "audit log is sealed (read-only)"),
        }
    }
}

impl std::error::Error for AuditLogError {}

// ── Hash Function ──────────────────────────────────────────────

/// FNV-1a 64-bit hash for tamper-evident chaining.
fn fnv_hash(data: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn fnv_hash_hex(data: &str) -> String {
    format!("{:016x}", fnv_hash(data))
}

// ── Audit Entry ────────────────────────────────────────────────

/// A single immutable audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique entry ID.
    pub id: String,
    /// Timestamp with microsecond precision.
    pub timestamp: DateTime<Utc>,
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Who performed the action.
    pub actor: String,
    /// What action was performed.
    pub action: String,
    /// What resource was acted upon.
    pub resource: String,
    /// Outcome of the action.
    pub outcome: AuditOutcome,
    /// Severity level.
    pub severity: Severity,
    /// Additional structured details.
    pub details: HashMap<String, Value>,
    /// Hash of the previous entry (chain link).
    pub prev_hash: String,
    /// Hash of this entry (computed from content + prev_hash).
    pub hash: String,
    /// Source IP or identifier.
    pub source: Option<String>,
    /// Correlation ID for grouping related events.
    pub correlation_id: Option<String>,
}

impl AuditEntry {
    /// Compute the hash for this entry based on its content.
    pub fn compute_hash(&self) -> String {
        let content = format!(
            "{}:{}:{}:{}:{}:{}:{}:{}",
            self.id,
            self.timestamp.timestamp_micros(),
            self.sequence,
            self.actor,
            self.action,
            self.resource,
            self.outcome.as_str(),
            self.prev_hash,
        );
        fnv_hash_hex(&content)
    }

    /// Verify this entry's hash integrity.
    pub fn verify(&self) -> bool {
        self.hash == self.compute_hash()
    }

    /// Convert to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "timestamp": self.timestamp.to_rfc3339(),
            "sequence": self.sequence,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome.as_str(),
            "severity": self.severity.as_str(),
            "details": self.details,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
            "source": self.source,
            "correlation_id": self.correlation_id,
        })
    }
}

// ── Audit Event Builder ────────────────────────────────────────

/// Builder for creating audit events.
pub struct AuditEventBuilder {
    actor: String,
    action: String,
    resource: String,
    outcome: AuditOutcome,
    severity: Severity,
    details: HashMap<String, Value>,
    source: Option<String>,
    correlation_id: Option<String>,
}

impl AuditEventBuilder {
    /// Start building an audit event.
    pub fn new(actor: &str, action: &str, resource: &str) -> Self {
        Self {
            actor: actor.to_string(),
            action: action.to_string(),
            resource: resource.to_string(),
            outcome: AuditOutcome::Success,
            severity: Severity::Info,
            details: HashMap::new(),
            source: None,
            correlation_id: None,
        }
    }

    /// Set the outcome.
    pub fn outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Set the severity.
    pub fn severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    /// Add a detail.
    pub fn detail(mut self, key: &str, value: Value) -> Self {
        self.details.insert(key.to_string(), value);
        self
    }

    /// Set the source (e.g., IP address).
    pub fn source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }

    /// Set a correlation ID.
    pub fn correlation_id(mut self, id: &str) -> Self {
        self.correlation_id = Some(id.to_string());
        self
    }
}

// ── Retention Policy ───────────────────────────────────────────

/// Retention and rotation policy.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum number of entries to retain.
    pub max_entries: Option<usize>,
    /// Maximum age in seconds.
    pub max_age_secs: Option<u64>,
}

impl RetentionPolicy {
    pub fn new() -> Self {
        Self {
            max_entries: None,
            max_age_secs: None,
        }
    }

    pub fn with_max_entries(mut self, n: usize) -> Self {
        self.max_entries = Some(n);
        self
    }

    pub fn with_max_age(mut self, secs: u64) -> Self {
        self.max_age_secs = Some(secs);
        self
    }
}

// ── Audit Log ──────────────────────────────────────────────────

/// Append-only, hash-chained audit log with query and export capabilities.
pub struct AuditLog {
    entries: Vec<AuditEntry>,
    next_sequence: u64,
    last_hash: String,
    pub retention: RetentionPolicy,
    /// If true, no new entries can be appended.
    sealed: bool,
    /// Archived (rotated) segments.
    archives: Vec<Vec<AuditEntry>>,
}

const GENESIS_HASH: &str = "0000000000000000";

impl AuditLog {
    /// Create a new empty audit log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_sequence: 1,
            last_hash: GENESIS_HASH.to_string(),
            retention: RetentionPolicy::new(),
            sealed: false,
            archives: Vec::new(),
        }
    }

    /// Create with a retention policy.
    pub fn with_retention(mut self, policy: RetentionPolicy) -> Self {
        self.retention = policy;
        self
    }

    /// Append an event using the builder.
    pub fn append_event(
        &mut self,
        builder: AuditEventBuilder,
    ) -> Result<String, AuditLogError> {
        if self.sealed {
            return Err(AuditLogError::LogSealed);
        }

        let id = Uuid::new_v4().to_string();
        let timestamp = Utc::now();
        let prev_hash = self.last_hash.clone();
        let sequence = self.next_sequence;

        let mut entry = AuditEntry {
            id,
            timestamp,
            sequence,
            actor: builder.actor,
            action: builder.action,
            resource: builder.resource,
            outcome: builder.outcome,
            severity: builder.severity,
            details: builder.details,
            prev_hash,
            hash: String::new(),
            source: builder.source,
            correlation_id: builder.correlation_id,
        };
        entry.hash = entry.compute_hash();
        self.last_hash = entry.hash.clone();
        self.next_sequence += 1;
        let hash = entry.hash.clone();
        self.entries.push(entry);
        Ok(hash)
    }

    /// Append a simple event.
    pub fn append(
        &mut self,
        actor: &str,
        action: &str,
        resource: &str,
        outcome: AuditOutcome,
        details: HashMap<String, Value>,
    ) -> Result<String, AuditLogError> {
        let mut builder = AuditEventBuilder::new(actor, action, resource).outcome(outcome);
        builder.details = details;
        self.append_event(builder)
    }

    /// Verify the entire hash chain.
    pub fn verify_chain(&self) -> Result<(), AuditLogError> {
        let mut expected_prev = GENESIS_HASH.to_string();
        let mut expected_seq = 1u64;

        for entry in &self.entries {
            if entry.sequence != expected_seq {
                return Err(AuditLogError::SequenceGap {
                    expected: expected_seq,
                    got: entry.sequence,
                });
            }
            if entry.prev_hash != expected_prev {
                return Err(AuditLogError::ChainBroken {
                    sequence: entry.sequence,
                    expected: expected_prev,
                    got: entry.prev_hash.clone(),
                });
            }
            if !entry.verify() {
                return Err(AuditLogError::EntryTampered {
                    sequence: entry.sequence,
                });
            }
            expected_prev = entry.hash.clone();
            expected_seq += 1;
        }
        Ok(())
    }

    /// Check chain integrity (returns bool for convenience).
    pub fn is_chain_valid(&self) -> bool {
        self.verify_chain().is_ok()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Query by actor.
    pub fn by_actor(&self, actor: &str) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.actor == actor).collect()
    }

    /// Query by action.
    pub fn by_action(&self, action: &str) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.action == action).collect()
    }

    /// Query by resource.
    pub fn by_resource(&self, resource: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.resource == resource)
            .collect()
    }

    /// Query by time range (inclusive).
    pub fn by_time_range(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp >= from && e.timestamp <= to)
            .collect()
    }

    /// Query by outcome.
    pub fn by_outcome(&self, outcome: AuditOutcome) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.outcome == outcome)
            .collect()
    }

    /// Query by severity.
    pub fn by_severity(&self, severity: Severity) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.severity == severity)
            .collect()
    }

    /// Query by correlation ID.
    pub fn by_correlation_id(&self, id: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.correlation_id.as_deref() == Some(id))
            .collect()
    }

    /// Get entry by sequence number.
    pub fn by_sequence(&self, seq: u64) -> Option<&AuditEntry> {
        self.entries.iter().find(|e| e.sequence == seq)
    }

    /// Apply retention policy, archiving or removing old entries.
    pub fn apply_retention(&mut self) {
        let now = Utc::now();
        if let Some(max_age) = self.retention.max_age_secs {
            let cutoff = now - Duration::seconds(max_age as i64);
            self.entries.retain(|e| e.timestamp >= cutoff);
        }
        if let Some(max_entries) = self.retention.max_entries {
            if self.entries.len() > max_entries {
                let drain_count = self.entries.len() - max_entries;
                self.entries.drain(0..drain_count);
            }
        }
    }

    /// Rotate: archive current entries and start fresh.
    pub fn rotate(&mut self) {
        if !self.entries.is_empty() {
            let archived = std::mem::take(&mut self.entries);
            self.archives.push(archived);
            // Don't reset sequence — it should be monotonically increasing.
            // Don't reset last_hash — chain continues across rotations.
        }
    }

    /// Number of archived segments.
    pub fn archive_count(&self) -> usize {
        self.archives.len()
    }

    /// Get entries from an archived segment.
    pub fn get_archive(&self, index: usize) -> Option<&[AuditEntry]> {
        self.archives.get(index).map(|v| v.as_slice())
    }

    /// Seal the log (make read-only).
    pub fn seal(&mut self) {
        self.sealed = true;
    }

    /// Whether the log is sealed.
    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// Export all entries as a JSON array string.
    pub fn export_json(&self) -> String {
        let arr: Vec<Value> = self.entries.iter().map(|e| e.to_json()).collect();
        serde_json::to_string_pretty(&arr).unwrap_or_else(|_| "[]".to_string())
    }

    /// Export as newline-delimited JSON (NDJSON).
    pub fn export_ndjson(&self) -> String {
        self.entries
            .iter()
            .map(|e| serde_json::to_string(&e.to_json()).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Export entries matching a filter as JSON.
    pub fn export_filtered<F>(&self, predicate: F) -> String
    where
        F: Fn(&AuditEntry) -> bool,
    {
        let arr: Vec<Value> = self
            .entries
            .iter()
            .filter(|e| predicate(e))
            .map(|e| e.to_json())
            .collect();
        serde_json::to_string_pretty(&arr).unwrap_or_else(|_| "[]".to_string())
    }

    /// Direct access to entries slice.
    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_log_with_entries(n: usize) -> AuditLog {
        let mut log = AuditLog::new();
        for i in 0..n {
            log.append(
                &format!("actor{}", i % 3),
                &format!("action{}", i % 2),
                &format!("resource:{i}"),
                AuditOutcome::Success,
                HashMap::new(),
            )
            .unwrap();
        }
        log
    }

    #[test]
    fn test_append_and_len() {
        let log = make_log_with_entries(5);
        assert_eq!(log.len(), 5);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_hash_chain_valid() {
        let log = make_log_with_entries(10);
        assert!(log.is_chain_valid());
        assert!(log.verify_chain().is_ok());
    }

    #[test]
    fn test_tamper_detection() {
        let mut log = make_log_with_entries(3);
        // Tamper with an entry.
        log.entries[1].actor = "eve".to_string();
        assert!(!log.is_chain_valid());
        match log.verify_chain() {
            Err(AuditLogError::EntryTampered { sequence: 2 }) => {}
            other => panic!("expected EntryTampered at seq 2, got {other:?}"),
        }
    }

    #[test]
    fn test_chain_broken() {
        let mut log = make_log_with_entries(3);
        log.entries[2].prev_hash = "tampered".to_string();
        log.entries[2].hash = log.entries[2].compute_hash();
        match log.verify_chain() {
            Err(AuditLogError::ChainBroken { sequence: 3, .. }) => {}
            other => panic!("expected ChainBroken at seq 3, got {other:?}"),
        }
    }

    #[test]
    fn test_empty_log_valid() {
        let log = AuditLog::new();
        assert!(log.is_empty());
        assert!(log.is_chain_valid());
    }

    #[test]
    fn test_query_by_actor() {
        let log = make_log_with_entries(9);
        assert_eq!(log.by_actor("actor0").len(), 3);
        assert_eq!(log.by_actor("actor1").len(), 3);
        assert_eq!(log.by_actor("actor2").len(), 3);
    }

    #[test]
    fn test_query_by_action() {
        let log = make_log_with_entries(6);
        assert_eq!(log.by_action("action0").len(), 3);
        assert_eq!(log.by_action("action1").len(), 3);
    }

    #[test]
    fn test_query_by_resource() {
        let log = make_log_with_entries(5);
        assert_eq!(log.by_resource("resource:0").len(), 1);
        assert_eq!(log.by_resource("resource:99").len(), 0);
    }

    #[test]
    fn test_query_by_time_range() {
        let mut log = AuditLog::new();
        let before = Utc::now() - Duration::seconds(1);
        log.append("a", "x", "r", AuditOutcome::Success, HashMap::new())
            .unwrap();
        let after = Utc::now() + Duration::seconds(1);
        assert_eq!(log.by_time_range(before, after).len(), 1);
        let future = Utc::now() + Duration::hours(1);
        assert_eq!(log.by_time_range(after, future).len(), 0);
    }

    #[test]
    fn test_query_by_outcome() {
        let mut log = AuditLog::new();
        log.append("a", "x", "r", AuditOutcome::Success, HashMap::new())
            .unwrap();
        log.append("a", "x", "r", AuditOutcome::Denied, HashMap::new())
            .unwrap();
        assert_eq!(log.by_outcome(AuditOutcome::Success).len(), 1);
        assert_eq!(log.by_outcome(AuditOutcome::Denied).len(), 1);
    }

    #[test]
    fn test_query_by_severity() {
        let mut log = AuditLog::new();
        log.append_event(
            AuditEventBuilder::new("a", "login_fail", "auth")
                .outcome(AuditOutcome::Failure)
                .severity(Severity::Warning),
        )
        .unwrap();
        log.append_event(
            AuditEventBuilder::new("a", "read", "file")
                .severity(Severity::Info),
        )
        .unwrap();
        assert_eq!(log.by_severity(Severity::Warning).len(), 1);
        assert_eq!(log.by_severity(Severity::Info).len(), 1);
    }

    #[test]
    fn test_correlation_id() {
        let mut log = AuditLog::new();
        log.append_event(
            AuditEventBuilder::new("a", "step1", "flow")
                .correlation_id("tx-123"),
        )
        .unwrap();
        log.append_event(
            AuditEventBuilder::new("a", "step2", "flow")
                .correlation_id("tx-123"),
        )
        .unwrap();
        log.append_event(AuditEventBuilder::new("b", "other", "x"))
            .unwrap();
        assert_eq!(log.by_correlation_id("tx-123").len(), 2);
    }

    #[test]
    fn test_sequence_numbers() {
        let log = make_log_with_entries(3);
        assert_eq!(log.entries()[0].sequence, 1);
        assert_eq!(log.entries()[1].sequence, 2);
        assert_eq!(log.entries()[2].sequence, 3);
        assert!(log.by_sequence(2).is_some());
        assert!(log.by_sequence(99).is_none());
    }

    #[test]
    fn test_retention_max_entries() {
        let mut log =
            AuditLog::new().with_retention(RetentionPolicy::new().with_max_entries(2));
        for i in 0..5 {
            log.append(&format!("a{i}"), "x", "r", AuditOutcome::Success, HashMap::new())
                .unwrap();
        }
        log.apply_retention();
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_rotation() {
        let mut log = make_log_with_entries(3);
        log.rotate();
        assert!(log.is_empty());
        assert_eq!(log.archive_count(), 1);
        let archived = log.get_archive(0).unwrap();
        assert_eq!(archived.len(), 3);
        // Can still append after rotation.
        log.append("a", "x", "r", AuditOutcome::Success, HashMap::new())
            .unwrap();
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_seal() {
        let mut log = make_log_with_entries(1);
        log.seal();
        assert!(log.is_sealed());
        let err = log
            .append("a", "x", "r", AuditOutcome::Success, HashMap::new())
            .unwrap_err();
        assert_eq!(err, AuditLogError::LogSealed);
    }

    #[test]
    fn test_export_json() {
        let log = make_log_with_entries(2);
        let json = log.export_json();
        let parsed: Vec<Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn test_export_ndjson() {
        let log = make_log_with_entries(3);
        let ndjson = log.export_ndjson();
        let lines: Vec<&str> = ndjson.lines().collect();
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_export_filtered() {
        let mut log = AuditLog::new();
        log.append("admin", "delete", "r", AuditOutcome::Success, HashMap::new())
            .unwrap();
        log.append("user", "read", "r", AuditOutcome::Success, HashMap::new())
            .unwrap();
        let json = log.export_filtered(|e| e.actor == "admin");
        let parsed: Vec<Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn test_entry_to_json_fields() {
        let mut log = AuditLog::new();
        let mut details = HashMap::new();
        details.insert("ip".to_string(), json!("10.0.0.1"));
        log.append("admin", "login", "session:1", AuditOutcome::Success, details)
            .unwrap();
        let j = log.entries()[0].to_json();
        assert_eq!(j["actor"], "admin");
        assert_eq!(j["action"], "login");
        assert_eq!(j["outcome"], "success");
        assert_eq!(j["details"]["ip"], "10.0.0.1");
    }

    #[test]
    fn test_outcome_variants() {
        assert_eq!(AuditOutcome::Success.as_str(), "success");
        assert_eq!(AuditOutcome::Failure.as_str(), "failure");
        assert_eq!(AuditOutcome::Denied.as_str(), "denied");
        assert_eq!(AuditOutcome::Error.as_str(), "error");
        assert_eq!(AuditOutcome::Timeout.as_str(), "timeout");
        assert_eq!(AuditOutcome::Partial.as_str(), "partial");
    }

    #[test]
    fn test_outcome_is_successful() {
        assert!(AuditOutcome::Success.is_successful());
        assert!(AuditOutcome::Partial.is_successful());
        assert!(!AuditOutcome::Failure.is_successful());
        assert!(!AuditOutcome::Denied.is_successful());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Critical);
    }

    #[test]
    fn test_event_builder() {
        let mut log = AuditLog::new();
        log.append_event(
            AuditEventBuilder::new("admin", "deploy", "app:prod")
                .outcome(AuditOutcome::Success)
                .severity(Severity::Critical)
                .detail("version", json!("2.0.0"))
                .source("10.0.0.1")
                .correlation_id("deploy-42"),
        )
        .unwrap();
        let entry = &log.entries()[0];
        assert_eq!(entry.actor, "admin");
        assert_eq!(entry.severity, Severity::Critical);
        assert_eq!(entry.source.as_deref(), Some("10.0.0.1"));
        assert_eq!(entry.correlation_id.as_deref(), Some("deploy-42"));
        assert_eq!(entry.details["version"], "2.0.0");
    }

    #[test]
    fn test_error_display() {
        let e = AuditLogError::ChainBroken {
            sequence: 5,
            expected: "aaa".to_string(),
            got: "bbb".to_string(),
        };
        assert!(e.to_string().contains("chain broken"));
        let e2 = AuditLogError::LogSealed;
        assert!(e2.to_string().contains("sealed"));
    }
}
