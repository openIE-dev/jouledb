//! Structured security audit log for authentication and authorization events.
//!
//! Every security-relevant action (login, token issuance, secret access, etc.)
//! is recorded as an [`AuditEntry`] inside an [`AuditLog`].  The log is
//! bounded: once `max_entries` is reached the oldest entries are evicted, but
//! the monotonic [`AuditLog::total_logged`] counter keeps incrementing so
//! operators can detect gaps.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Security-relevant actions that are recorded in the audit log.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditAction {
    Login,
    LoginFailed,
    Logout,
    TokenIssued,
    TokenRefreshed,
    TokenRevoked,
    PermissionGranted,
    PermissionDenied,
    ApiKeyCreated,
    ApiKeyRevoked,
    SecretAccessed,
    SecretCreated,
    SecretRotated,
    SecretDeleted,
    RoleChanged,
    SessionCreated,
    SessionTerminated,
}

/// Outcome of an audited action.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditOutcome {
    Success,
    Failure,
    Denied,
}

// ---------------------------------------------------------------------------
// Entry & Query
// ---------------------------------------------------------------------------

/// A single audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique identifier for this entry.
    pub id: String,
    /// When the event occurred.
    pub timestamp: DateTime<Utc>,
    /// What action was performed.
    pub action: AuditAction,
    /// Who performed the action (user id, service account, etc.).
    pub actor: String,
    /// Organisation scope.
    pub org: String,
    /// The resource that was acted upon (token jti, secret name, etc.).
    pub resource: String,
    /// Whether the action succeeded, failed, or was denied.
    pub outcome: AuditOutcome,
    /// Optional source IP address.
    pub ip_addr: Option<String>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

/// Filter parameters for querying the audit log.
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    pub actor: Option<String>,
    pub org: Option<String>,
    pub action: Option<AuditAction>,
    pub outcome: Option<AuditOutcome>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
}

// ---------------------------------------------------------------------------
// AuditLog
// ---------------------------------------------------------------------------

/// Callback invoked after each audit entry is recorded.
///
/// Implementations should persist the entry to durable storage (TSDB, disk,
/// remote log aggregator) so that audit data survives process restarts.
///
/// Regulatory basis: AU-9 (Protection of Audit Information) — audit records
/// must be written to durable media to survive power loss.
pub trait AuditSink: Send + Sync {
    /// Persist an audit entry to durable storage.
    ///
    /// Errors are logged but do not prevent in-memory recording — the audit
    /// pipeline must not block on I/O.
    fn persist(&self, entry: &AuditEntry);
}

/// Bounded, in-memory audit log with optional durable sink.
///
/// When a [`AuditSink`] is attached, every recorded entry is also forwarded
/// to the sink for durable persistence (AU-9, PCI DSS 10.2).
pub struct AuditLog {
    entries: Arc<RwLock<Vec<AuditEntry>>>,
    max_entries: usize,
    total_logged: AtomicU64,
    sink: RwLock<Option<Arc<dyn AuditSink>>>,
}

impl AuditLog {
    /// Create a new audit log that retains at most `max_entries` entries.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(Vec::with_capacity(max_entries.min(4096)))),
            max_entries,
            total_logged: AtomicU64::new(0),
            sink: RwLock::new(None),
        }
    }

    /// Attach a durable sink for persistent audit log storage.
    ///
    /// Once attached, every subsequent `record()` call forwards the entry
    /// to the sink. This satisfies AU-9 (Protection of Audit Information)
    /// and PCI DSS 10.2 (12-month audit retention).
    pub fn set_sink(&self, sink: Arc<dyn AuditSink>) {
        *self.sink.write().unwrap() = Some(sink);
    }

    /// Record an arbitrary [`AuditEntry`].  Returns the entry id.
    ///
    /// If the log has reached its capacity the oldest entry is evicted.
    /// When a durable sink is attached, the entry is also persisted.
    pub fn record(&self, mut entry: AuditEntry) -> String {
        if entry.id.is_empty() {
            entry.id = Uuid::new_v4().to_string();
        }
        let id = entry.id.clone();

        // Forward to durable sink before in-memory storage (AU-9)
        if let Some(sink) = self.sink.read().unwrap().as_ref() {
            sink.persist(&entry);
        }

        let mut entries = self.entries.write().unwrap();
        if entries.len() >= self.max_entries {
            entries.remove(0);
        }
        entries.push(entry);
        self.total_logged.fetch_add(1, Ordering::Relaxed);

        id
    }

    /// Convenience: record a successful action.
    pub fn record_success(
        &self,
        action: AuditAction,
        actor: impl Into<String>,
        org: impl Into<String>,
        resource: impl Into<String>,
    ) -> String {
        self.record(AuditEntry {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            action,
            actor: actor.into(),
            org: org.into(),
            resource: resource.into(),
            outcome: AuditOutcome::Success,
            ip_addr: None,
            metadata: HashMap::new(),
        })
    }

    /// Convenience: record a failed action, attaching the `reason` in metadata.
    pub fn record_failure(
        &self,
        action: AuditAction,
        actor: impl Into<String>,
        org: impl Into<String>,
        resource: impl Into<String>,
        reason: impl Into<String>,
    ) -> String {
        let mut metadata = HashMap::new();
        metadata.insert("reason".into(), reason.into());

        self.record(AuditEntry {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            action,
            actor: actor.into(),
            org: org.into(),
            resource: resource.into(),
            outcome: AuditOutcome::Failure,
            ip_addr: None,
            metadata,
        })
    }

    /// Query the log with optional filters.  Results are returned in
    /// chronological order (oldest first).
    pub fn query(&self, q: &AuditQuery) -> Vec<AuditEntry> {
        let entries = self.entries.read().unwrap();
        let iter = entries.iter().filter(|e| {
            if let Some(ref actor) = q.actor
                && e.actor != *actor
            {
                return false;
            }
            if let Some(ref org) = q.org
                && e.org != *org
            {
                return false;
            }
            if let Some(action) = q.action
                && e.action != action
            {
                return false;
            }
            if let Some(outcome) = q.outcome
                && e.outcome != outcome
            {
                return false;
            }
            if let Some(since) = q.since
                && e.timestamp < since
            {
                return false;
            }
            if let Some(until) = q.until
                && e.timestamp > until
            {
                return false;
            }
            true
        });

        match q.limit {
            Some(limit) => iter.take(limit).cloned().collect(),
            None => iter.cloned().collect(),
        }
    }

    /// Return all entries in chronological order.
    pub fn all(&self) -> Vec<AuditEntry> {
        self.entries.read().unwrap().clone()
    }

    /// Return the `limit` most recent entries, newest first.
    pub fn recent(&self, limit: usize) -> Vec<AuditEntry> {
        let entries = self.entries.read().unwrap();
        entries.iter().rev().take(limit).cloned().collect()
    }

    /// Total number of entries ever recorded, including evicted ones.
    pub fn total_logged(&self) -> u64 {
        self.total_logged.load(Ordering::Relaxed)
    }

    /// Number of entries currently retained in the log.
    pub fn count(&self) -> usize {
        self.entries.read().unwrap().len()
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new(50_000)
    }
}

/// Whether a durable sink has been attached.
impl AuditLog {
    pub fn has_sink(&self) -> bool {
        self.sink.read().unwrap().is_some()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn make_entry(action: AuditAction, actor: &str, org: &str) -> AuditEntry {
        AuditEntry {
            id: String::new(),
            timestamp: Utc::now(),
            action,
            actor: actor.into(),
            org: org.into(),
            resource: "res".into(),
            outcome: AuditOutcome::Success,
            ip_addr: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn record_and_query_all() {
        let log = AuditLog::new(100);
        log.record(make_entry(AuditAction::Login, "alice", "acme"));
        log.record(make_entry(AuditAction::Logout, "bob", "acme"));

        let all = log.all();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].actor, "alice");
        assert_eq!(all[1].actor, "bob");
    }

    #[test]
    fn filter_by_actor() {
        let log = AuditLog::new(100);
        log.record(make_entry(AuditAction::Login, "alice", "acme"));
        log.record(make_entry(AuditAction::Login, "bob", "acme"));
        log.record(make_entry(AuditAction::Logout, "alice", "acme"));

        let q = AuditQuery {
            actor: Some("alice".into()),
            ..Default::default()
        };
        let results = log.query(&q);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|e| e.actor == "alice"));
    }

    #[test]
    fn filter_by_org() {
        let log = AuditLog::new(100);
        log.record(make_entry(AuditAction::Login, "alice", "acme"));
        log.record(make_entry(AuditAction::Login, "alice", "globex"));

        let q = AuditQuery {
            org: Some("globex".into()),
            ..Default::default()
        };
        let results = log.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].org, "globex");
    }

    #[test]
    fn filter_by_action() {
        let log = AuditLog::new(100);
        log.record(make_entry(AuditAction::Login, "alice", "acme"));
        log.record(make_entry(AuditAction::TokenIssued, "alice", "acme"));
        log.record(make_entry(AuditAction::Login, "bob", "acme"));

        let q = AuditQuery {
            action: Some(AuditAction::Login),
            ..Default::default()
        };
        let results = log.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn filter_by_outcome() {
        let log = AuditLog::new(100);
        log.record_success(AuditAction::Login, "alice", "acme", "session-1");
        log.record_failure(
            AuditAction::Login,
            "eve",
            "acme",
            "session-2",
            "bad password",
        );

        let q = AuditQuery {
            outcome: Some(AuditOutcome::Failure),
            ..Default::default()
        };
        let results = log.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].actor, "eve");
        assert_eq!(results[0].metadata.get("reason").unwrap(), "bad password");
    }

    #[test]
    fn filter_by_time_range() {
        let log = AuditLog::new(100);
        let now = Utc::now();

        let mut old = make_entry(AuditAction::Login, "alice", "acme");
        old.timestamp = now - Duration::hours(2);
        log.record(old);

        let mut recent = make_entry(AuditAction::Logout, "alice", "acme");
        recent.timestamp = now;
        log.record(recent);

        let q = AuditQuery {
            since: Some(now - Duration::hours(1)),
            ..Default::default()
        };
        let results = log.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].action, AuditAction::Logout);

        let q2 = AuditQuery {
            until: Some(now - Duration::hours(1)),
            ..Default::default()
        };
        let results2 = log.query(&q2);
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].action, AuditAction::Login);
    }

    #[test]
    fn eviction_when_full() {
        let log = AuditLog::new(3);
        log.record(make_entry(AuditAction::Login, "a", "o"));
        log.record(make_entry(AuditAction::Login, "b", "o"));
        log.record(make_entry(AuditAction::Login, "c", "o"));
        assert_eq!(log.count(), 3);

        // Fourth entry evicts the oldest ("a").
        log.record(make_entry(AuditAction::Login, "d", "o"));
        assert_eq!(log.count(), 3);

        let all = log.all();
        assert_eq!(all[0].actor, "b");
        assert_eq!(all[2].actor, "d");
    }

    #[test]
    fn recent_returns_newest_first() {
        let log = AuditLog::new(100);
        log.record(make_entry(AuditAction::Login, "first", "o"));
        log.record(make_entry(AuditAction::Login, "second", "o"));
        log.record(make_entry(AuditAction::Login, "third", "o"));

        let r = log.recent(2);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].actor, "third");
        assert_eq!(r[1].actor, "second");
    }

    #[test]
    fn total_logged_increments_after_eviction() {
        let log = AuditLog::new(2);
        log.record(make_entry(AuditAction::Login, "a", "o"));
        log.record(make_entry(AuditAction::Login, "b", "o"));
        log.record(make_entry(AuditAction::Login, "c", "o"));
        log.record(make_entry(AuditAction::Login, "d", "o"));

        // Only 2 retained, but 4 total logged.
        assert_eq!(log.count(), 2);
        assert_eq!(log.total_logged(), 4);
    }

    #[test]
    fn record_success_convenience() {
        let log = AuditLog::new(100);
        let id = log.record_success(AuditAction::TokenIssued, "svc", "acme", "tok-123");
        assert!(!id.is_empty());

        let all = log.all();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].outcome, AuditOutcome::Success);
        assert_eq!(all[0].resource, "tok-123");
    }

    #[test]
    fn record_failure_convenience() {
        let log = AuditLog::new(100);
        let id = log.record_failure(
            AuditAction::LoginFailed,
            "eve",
            "acme",
            "session",
            "invalid password",
        );
        assert!(!id.is_empty());

        let all = log.all();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].outcome, AuditOutcome::Failure);
        assert_eq!(all[0].metadata.get("reason").unwrap(), "invalid password",);
    }

    #[test]
    fn audit_action_serde_roundtrip() {
        let action = AuditAction::SecretRotated;
        let json = serde_json::to_string(&action).unwrap();
        assert_eq!(json, "\"secret_rotated\"");

        let back: AuditAction = serde_json::from_str(&json).unwrap();
        assert_eq!(back, action);
    }

    #[test]
    fn audit_outcome_serde_roundtrip() {
        let outcome = AuditOutcome::Denied;
        let json = serde_json::to_string(&outcome).unwrap();
        assert_eq!(json, "\"denied\"");

        let back: AuditOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(back, outcome);
    }

    #[test]
    fn default_impl() {
        let log = AuditLog::default();
        assert_eq!(log.count(), 0);
        assert_eq!(log.total_logged(), 0);
        // Default capacity is 50_000.
        assert_eq!(log.max_entries, 50_000);
        assert!(!log.has_sink());
    }

    #[test]
    fn sink_receives_entries() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingSink(AtomicUsize);
        impl AuditSink for CountingSink {
            fn persist(&self, _entry: &AuditEntry) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let log = AuditLog::new(100);
        let sink = Arc::new(CountingSink(AtomicUsize::new(0)));
        log.set_sink(sink.clone());
        assert!(log.has_sink());

        log.record_success(AuditAction::Login, "alice", "acme", "session-1");
        log.record_failure(
            AuditAction::LoginFailed,
            "eve",
            "acme",
            "session-2",
            "bad pw",
        );

        assert_eq!(sink.0.load(Ordering::Relaxed), 2);
        assert_eq!(log.count(), 2);
    }
}
