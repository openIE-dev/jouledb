//! Audit trail — immutable event log, actor/action/resource/outcome,
//! tamper detection via hash chain, query by time range/actor/resource,
//! retention policies.
//!
//! Pure-Rust replacement for audit logging libraries and compliance trail systems.

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

// ── Audit event ──────────────────────────────────────────────────

/// A single audit event recording who did what to which resource and the outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditEvent {
    /// Monotonic sequence number.
    pub sequence: u64,
    /// Timestamp in seconds since epoch.
    pub timestamp_s: u64,
    /// Who performed the action.
    pub actor: String,
    /// What action was performed.
    pub action: String,
    /// Which resource was affected.
    pub resource: String,
    /// The outcome of the action.
    pub outcome: Outcome,
    /// Optional details or metadata.
    pub details: String,
    /// Hash of the previous event in the chain (0 for first event).
    pub prev_hash: u64,
    /// Hash of this event (computed from all fields + prev_hash).
    pub event_hash: u64,
}

/// Outcome of an audited action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Outcome {
    Success,
    Failure,
    Denied,
    Error,
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::Success => write!(f, "success"),
            Outcome::Failure => write!(f, "failure"),
            Outcome::Denied => write!(f, "denied"),
            Outcome::Error => write!(f, "error"),
        }
    }
}

/// Compute a deterministic hash for an audit event's content.
fn compute_event_hash(
    sequence: u64,
    timestamp_s: u64,
    actor: &str,
    action: &str,
    resource: &str,
    outcome: &Outcome,
    details: &str,
    prev_hash: u64,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    sequence.hash(&mut hasher);
    timestamp_s.hash(&mut hasher);
    actor.hash(&mut hasher);
    action.hash(&mut hasher);
    resource.hash(&mut hasher);
    outcome.hash(&mut hasher);
    details.hash(&mut hasher);
    prev_hash.hash(&mut hasher);
    hasher.finish()
}

// ── Retention policy ─────────────────────────────────────────────

/// Retention policy for audit events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetentionPolicy {
    /// Maximum age in seconds for events to be retained.
    pub max_age_seconds: Option<u64>,
    /// Maximum number of events to retain.
    pub max_count: Option<usize>,
}

impl RetentionPolicy {
    pub fn by_age(seconds: u64) -> Self {
        Self { max_age_seconds: Some(seconds), max_count: None }
    }

    pub fn by_count(count: usize) -> Self {
        Self { max_age_seconds: None, max_count: Some(count) }
    }

    pub fn both(max_age_seconds: u64, max_count: usize) -> Self {
        Self { max_age_seconds: Some(max_age_seconds), max_count: Some(max_count) }
    }

    pub fn unlimited() -> Self {
        Self { max_age_seconds: None, max_count: None }
    }
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self::unlimited()
    }
}

// ── Query filter ─────────────────────────────────────────────────

/// Filter for querying audit events.
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    pub start_time_s: Option<u64>,
    pub end_time_s: Option<u64>,
    pub actor: Option<String>,
    pub action: Option<String>,
    pub resource: Option<String>,
    pub outcome: Option<Outcome>,
    pub limit: Option<usize>,
}

impl AuditQuery {
    pub fn new() -> Self { Self::default() }

    pub fn with_time_range(mut self, start: u64, end: u64) -> Self {
        self.start_time_s = Some(start);
        self.end_time_s = Some(end);
        self
    }

    pub fn with_actor(mut self, actor: impl Into<String>) -> Self {
        self.actor = Some(actor.into());
        self
    }

    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.action = Some(action.into());
        self
    }

    pub fn with_resource(mut self, resource: impl Into<String>) -> Self {
        self.resource = Some(resource.into());
        self
    }

    pub fn with_outcome(mut self, outcome: Outcome) -> Self {
        self.outcome = Some(outcome);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Check if an event matches this query.
    fn matches(&self, event: &AuditEvent) -> bool {
        if let Some(start) = self.start_time_s {
            if event.timestamp_s < start { return false; }
        }
        if let Some(end) = self.end_time_s {
            if event.timestamp_s > end { return false; }
        }
        if let Some(actor) = &self.actor {
            if event.actor != *actor { return false; }
        }
        if let Some(action) = &self.action {
            if event.action != *action { return false; }
        }
        if let Some(resource) = &self.resource {
            if event.resource != *resource { return false; }
        }
        if let Some(outcome) = &self.outcome {
            if event.outcome != *outcome { return false; }
        }
        true
    }
}

// ── Verification result ──────────────────────────────────────────

/// Result of verifying the audit trail integrity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationResult {
    pub is_valid: bool,
    pub events_checked: usize,
    /// Index of first tampered event, if any.
    pub first_tampered_index: Option<usize>,
    pub error_message: Option<String>,
}

// ── Audit Trail ──────────────────────────────────────────────────

/// Immutable append-only audit trail with hash chain integrity.
#[derive(Debug, Clone)]
pub struct AuditTrail {
    events: Vec<AuditEvent>,
    next_sequence: u64,
    last_hash: u64,
    retention: RetentionPolicy,
}

impl AuditTrail {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            next_sequence: 1,
            last_hash: 0,
            retention: RetentionPolicy::default(),
        }
    }

    pub fn with_retention(mut self, policy: RetentionPolicy) -> Self {
        self.retention = policy;
        self
    }

    /// Append an event to the audit trail. Returns the sequence number.
    pub fn append(
        &mut self,
        timestamp_s: u64,
        actor: impl Into<String>,
        action: impl Into<String>,
        resource: impl Into<String>,
        outcome: Outcome,
        details: impl Into<String>,
    ) -> u64 {
        let seq = self.next_sequence;
        let actor = actor.into();
        let action = action.into();
        let resource = resource.into();
        let details = details.into();

        let event_hash = compute_event_hash(
            seq, timestamp_s, &actor, &action, &resource,
            &outcome, &details, self.last_hash,
        );

        let event = AuditEvent {
            sequence: seq,
            timestamp_s,
            actor,
            action,
            resource,
            outcome,
            details,
            prev_hash: self.last_hash,
            event_hash,
        };

        self.events.push(event);
        self.last_hash = event_hash;
        self.next_sequence += 1;
        seq
    }

    /// Query events matching the given filter.
    pub fn query(&self, filter: &AuditQuery) -> Vec<&AuditEvent> {
        let mut results: Vec<&AuditEvent> = self.events.iter()
            .filter(|e| filter.matches(e))
            .collect();

        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }
        results
    }

    /// Get all events.
    pub fn events(&self) -> &[AuditEvent] {
        &self.events
    }

    /// Get an event by sequence number.
    pub fn get_by_sequence(&self, sequence: u64) -> Option<&AuditEvent> {
        self.events.iter().find(|e| e.sequence == sequence)
    }

    /// Verify the integrity of the entire hash chain.
    pub fn verify_integrity(&self) -> VerificationResult {
        let mut prev_hash: u64 = 0;

        for (i, event) in self.events.iter().enumerate() {
            // Check that prev_hash matches.
            if event.prev_hash != prev_hash {
                return VerificationResult {
                    is_valid: false,
                    events_checked: i + 1,
                    first_tampered_index: Some(i),
                    error_message: Some(format!(
                        "Event {} prev_hash mismatch: expected {}, got {}",
                        event.sequence, prev_hash, event.prev_hash
                    )),
                };
            }

            // Recompute the event hash.
            let expected_hash = compute_event_hash(
                event.sequence, event.timestamp_s, &event.actor,
                &event.action, &event.resource, &event.outcome,
                &event.details, event.prev_hash,
            );

            if event.event_hash != expected_hash {
                return VerificationResult {
                    is_valid: false,
                    events_checked: i + 1,
                    first_tampered_index: Some(i),
                    error_message: Some(format!(
                        "Event {} hash mismatch: expected {}, got {}",
                        event.sequence, expected_hash, event.event_hash
                    )),
                };
            }

            prev_hash = event.event_hash;
        }

        VerificationResult {
            is_valid: true,
            events_checked: self.events.len(),
            first_tampered_index: None,
            error_message: None,
        }
    }

    /// Apply the retention policy, removing old events.
    /// Returns the number of events removed.
    pub fn apply_retention(&mut self, now_s: u64) -> usize {
        let original_len = self.events.len();

        // Apply age-based retention.
        if let Some(max_age) = self.retention.max_age_seconds {
            let cutoff = now_s.saturating_sub(max_age);
            self.events.retain(|e| e.timestamp_s > cutoff);
        }

        // Apply count-based retention.
        if let Some(max_count) = self.retention.max_count {
            if self.events.len() > max_count {
                let remove = self.events.len() - max_count;
                self.events.drain(..remove);
            }
        }

        original_len - self.events.len()
    }

    /// Total number of events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the trail is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// The last hash in the chain.
    pub fn chain_head(&self) -> u64 {
        self.last_hash
    }

    /// Count events by outcome.
    pub fn count_by_outcome(&self) -> Vec<(Outcome, usize)> {
        let mut success = 0usize;
        let mut failure = 0usize;
        let mut denied = 0usize;
        let mut error = 0usize;

        for event in &self.events {
            match event.outcome {
                Outcome::Success => success += 1,
                Outcome::Failure => failure += 1,
                Outcome::Denied => denied += 1,
                Outcome::Error => error += 1,
            }
        }

        let mut result = Vec::new();
        if success > 0 { result.push((Outcome::Success, success)); }
        if denied > 0 { result.push((Outcome::Denied, denied)); }
        if error > 0 { result.push((Outcome::Error, error)); }
        if failure > 0 { result.push((Outcome::Failure, failure)); }
        result
    }

    /// Get distinct actors in the trail.
    pub fn distinct_actors(&self) -> Vec<String> {
        let mut actors: Vec<String> = self.events.iter()
            .map(|e| e.actor.clone())
            .collect();
        actors.sort();
        actors.dedup();
        actors
    }

    /// Get distinct resources in the trail.
    pub fn distinct_resources(&self) -> Vec<String> {
        let mut resources: Vec<String> = self.events.iter()
            .map(|e| e.resource.clone())
            .collect();
        resources.sort();
        resources.dedup();
        resources
    }
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outcome_display() {
        assert_eq!(format!("{}", Outcome::Success), "success");
        assert_eq!(format!("{}", Outcome::Failure), "failure");
        assert_eq!(format!("{}", Outcome::Denied), "denied");
        assert_eq!(format!("{}", Outcome::Error), "error");
    }

    #[test]
    fn test_append_event() {
        let mut trail = AuditTrail::new();
        let seq = trail.append(1000, "admin", "create", "users/42", Outcome::Success, "Created user");
        assert_eq!(seq, 1);
        assert_eq!(trail.len(), 1);
    }

    #[test]
    fn test_sequence_increments() {
        let mut trail = AuditTrail::new();
        let s1 = trail.append(1000, "admin", "create", "users/1", Outcome::Success, "");
        let s2 = trail.append(1001, "admin", "create", "users/2", Outcome::Success, "");
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
    }

    #[test]
    fn test_hash_chain() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "admin", "create", "users/1", Outcome::Success, "");
        trail.append(1001, "admin", "update", "users/1", Outcome::Success, "");
        trail.append(1002, "admin", "delete", "users/1", Outcome::Success, "");

        let events = trail.events();
        // First event's prev_hash is 0.
        assert_eq!(events[0].prev_hash, 0);
        // Second event's prev_hash is first event's hash.
        assert_eq!(events[1].prev_hash, events[0].event_hash);
        // Third event's prev_hash is second event's hash.
        assert_eq!(events[2].prev_hash, events[1].event_hash);
    }

    #[test]
    fn test_verify_integrity_valid() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "admin", "create", "r1", Outcome::Success, "");
        trail.append(1001, "user", "read", "r1", Outcome::Success, "");
        trail.append(1002, "admin", "delete", "r1", Outcome::Success, "");

        let result = trail.verify_integrity();
        assert!(result.is_valid);
        assert_eq!(result.events_checked, 3);
        assert!(result.first_tampered_index.is_none());
    }

    #[test]
    fn test_verify_integrity_tampered() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "admin", "create", "r1", Outcome::Success, "");
        trail.append(1001, "user", "read", "r1", Outcome::Success, "");

        // Tamper with the second event.
        trail.events[1].action = "write".to_string();

        let result = trail.verify_integrity();
        assert!(!result.is_valid);
        assert_eq!(result.first_tampered_index, Some(1));
    }

    #[test]
    fn test_verify_empty_trail() {
        let trail = AuditTrail::new();
        let result = trail.verify_integrity();
        assert!(result.is_valid);
        assert_eq!(result.events_checked, 0);
    }

    #[test]
    fn test_query_by_time_range() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "a", "create", "r", Outcome::Success, "");
        trail.append(2000, "a", "update", "r", Outcome::Success, "");
        trail.append(3000, "a", "delete", "r", Outcome::Success, "");

        let q = AuditQuery::new().with_time_range(1500, 2500);
        let results = trail.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].action, "update");
    }

    #[test]
    fn test_query_by_actor() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "admin", "create", "r", Outcome::Success, "");
        trail.append(1001, "user", "read", "r", Outcome::Success, "");
        trail.append(1002, "admin", "update", "r", Outcome::Success, "");

        let q = AuditQuery::new().with_actor("admin");
        let results = trail.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_resource() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "a", "create", "users/1", Outcome::Success, "");
        trail.append(1001, "a", "create", "users/2", Outcome::Success, "");
        trail.append(1002, "a", "read", "users/1", Outcome::Success, "");

        let q = AuditQuery::new().with_resource("users/1");
        let results = trail.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_outcome() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "a", "create", "r", Outcome::Success, "");
        trail.append(1001, "a", "create", "r", Outcome::Denied, "");
        trail.append(1002, "a", "create", "r", Outcome::Success, "");

        let q = AuditQuery::new().with_outcome(Outcome::Denied);
        let results = trail.query(&q);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_by_action() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "a", "create", "r", Outcome::Success, "");
        trail.append(1001, "a", "read", "r", Outcome::Success, "");

        let q = AuditQuery::new().with_action("read");
        let results = trail.query(&q);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_with_limit() {
        let mut trail = AuditTrail::new();
        for i in 0..10 {
            trail.append(1000 + i, "a", "act", "r", Outcome::Success, "");
        }
        let q = AuditQuery::new().with_limit(3);
        let results = trail.query(&q);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_query_combined_filters() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "admin", "create", "r1", Outcome::Success, "");
        trail.append(1001, "admin", "delete", "r1", Outcome::Failure, "");
        trail.append(1002, "user", "create", "r2", Outcome::Success, "");
        trail.append(1003, "admin", "create", "r1", Outcome::Success, "");

        let q = AuditQuery::new()
            .with_actor("admin")
            .with_action("create")
            .with_outcome(Outcome::Success);
        let results = trail.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_retention_by_age() {
        let mut trail = AuditTrail::new()
            .with_retention(RetentionPolicy::by_age(500));
        trail.append(1000, "a", "act", "r", Outcome::Success, "");
        trail.append(1200, "a", "act", "r", Outcome::Success, "");
        trail.append(1400, "a", "act", "r", Outcome::Success, "");

        let removed = trail.apply_retention(1500);
        assert_eq!(removed, 1);
        assert_eq!(trail.len(), 2);
    }

    #[test]
    fn test_retention_by_count() {
        let mut trail = AuditTrail::new()
            .with_retention(RetentionPolicy::by_count(2));
        trail.append(1000, "a", "act", "r", Outcome::Success, "");
        trail.append(1001, "a", "act", "r", Outcome::Success, "");
        trail.append(1002, "a", "act", "r", Outcome::Success, "");

        let removed = trail.apply_retention(2000);
        assert_eq!(removed, 1);
        assert_eq!(trail.len(), 2);
    }

    #[test]
    fn test_retention_both() {
        let mut trail = AuditTrail::new()
            .with_retention(RetentionPolicy::both(100, 5));
        for i in 0..10 {
            trail.append(1000 + i * 20, "a", "act", "r", Outcome::Success, "");
        }
        let removed = trail.apply_retention(1200);
        // Age removes some, then count trims further.
        assert!(trail.len() <= 5);
        assert!(removed > 0);
    }

    #[test]
    fn test_retention_unlimited() {
        let mut trail = AuditTrail::new()
            .with_retention(RetentionPolicy::unlimited());
        for i in 0..100 {
            trail.append(i, "a", "act", "r", Outcome::Success, "");
        }
        let removed = trail.apply_retention(200);
        assert_eq!(removed, 0);
        assert_eq!(trail.len(), 100);
    }

    #[test]
    fn test_get_by_sequence() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "admin", "create", "r1", Outcome::Success, "details");
        let event = trail.get_by_sequence(1).unwrap();
        assert_eq!(event.actor, "admin");
        assert!(trail.get_by_sequence(99).is_none());
    }

    #[test]
    fn test_count_by_outcome() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "a", "act", "r", Outcome::Success, "");
        trail.append(1001, "a", "act", "r", Outcome::Success, "");
        trail.append(1002, "a", "act", "r", Outcome::Denied, "");

        let counts = trail.count_by_outcome();
        let success_count = counts.iter().find(|(o, _)| *o == Outcome::Success).map(|(_, c)| *c).unwrap_or(0);
        assert_eq!(success_count, 2);
        let denied_count = counts.iter().find(|(o, _)| *o == Outcome::Denied).map(|(_, c)| *c).unwrap_or(0);
        assert_eq!(denied_count, 1);
    }

    #[test]
    fn test_distinct_actors() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "admin", "create", "r", Outcome::Success, "");
        trail.append(1001, "user", "read", "r", Outcome::Success, "");
        trail.append(1002, "admin", "update", "r", Outcome::Success, "");

        let actors = trail.distinct_actors();
        assert_eq!(actors, vec!["admin", "user"]);
    }

    #[test]
    fn test_distinct_resources() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "a", "act", "r1", Outcome::Success, "");
        trail.append(1001, "a", "act", "r2", Outcome::Success, "");
        trail.append(1002, "a", "act", "r1", Outcome::Success, "");

        let resources = trail.distinct_resources();
        assert_eq!(resources, vec!["r1", "r2"]);
    }

    #[test]
    fn test_chain_head() {
        let mut trail = AuditTrail::new();
        assert_eq!(trail.chain_head(), 0);
        trail.append(1000, "a", "act", "r", Outcome::Success, "");
        assert_ne!(trail.chain_head(), 0);
    }

    #[test]
    fn test_is_empty() {
        let mut trail = AuditTrail::new();
        assert!(trail.is_empty());
        trail.append(1000, "a", "act", "r", Outcome::Success, "");
        assert!(!trail.is_empty());
    }

    #[test]
    fn test_default_trail() {
        let trail = AuditTrail::default();
        assert!(trail.is_empty());
    }

    #[test]
    fn test_tamper_chain_break() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "a", "create", "r", Outcome::Success, "");
        trail.append(1001, "a", "read", "r", Outcome::Success, "");
        trail.append(1002, "a", "update", "r", Outcome::Success, "");

        // Break the chain by modifying prev_hash.
        trail.events[2].prev_hash = 12345;

        let result = trail.verify_integrity();
        assert!(!result.is_valid);
        assert_eq!(result.first_tampered_index, Some(2));
        assert!(result.error_message.unwrap().contains("prev_hash mismatch"));
    }

    #[test]
    fn test_retention_policy_default() {
        let policy = RetentionPolicy::default();
        assert!(policy.max_age_seconds.is_none());
        assert!(policy.max_count.is_none());
    }

    #[test]
    fn test_query_empty_trail() {
        let trail = AuditTrail::new();
        let q = AuditQuery::new();
        assert!(trail.query(&q).is_empty());
    }

    #[test]
    fn test_event_details_stored() {
        let mut trail = AuditTrail::new();
        trail.append(1000, "admin", "create", "users/42", Outcome::Success, "IP: 10.0.0.1");
        let event = trail.get_by_sequence(1).unwrap();
        assert_eq!(event.details, "IP: 10.0.0.1");
    }
}
