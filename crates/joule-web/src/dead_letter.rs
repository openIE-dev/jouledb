//! Dead letter queue — failed message capture, failure reason tracking,
//! retry from DLQ, max retry count, age-based expiry, and DLQ monitoring.
//!
//! Replaces JS dead-letter libraries (BullMQ dead letters, SQS DLQ) with
//! a pure-Rust dead letter queue that tracks energy per operation.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// DLQ errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DlqError {
    /// Message not found.
    MessageNotFound(String),
    /// Max retries exceeded.
    MaxRetriesExceeded { message_id: String, max_retries: u32 },
    /// Message already expired.
    MessageExpired(String),
    /// Queue not found.
    QueueNotFound(String),
    /// Duplicate message id.
    DuplicateMessage(String),
    /// Queue is empty.
    QueueEmpty(String),
}

impl std::fmt::Display for DlqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MessageNotFound(id) => write!(f, "message not found: {id}"),
            Self::MaxRetriesExceeded { message_id, max_retries } => {
                write!(f, "max retries ({max_retries}) exceeded for {message_id}")
            }
            Self::MessageExpired(id) => write!(f, "message expired: {id}"),
            Self::QueueNotFound(id) => write!(f, "queue not found: {id}"),
            Self::DuplicateMessage(id) => write!(f, "duplicate message: {id}"),
            Self::QueueEmpty(id) => write!(f, "queue is empty: {id}"),
        }
    }
}

impl std::error::Error for DlqError {}

// ── Message State ───────────────────────────────────────────────

/// Current state of a dead letter message.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DlqMessageState {
    /// Waiting in the DLQ for retry or manual review.
    Pending,
    /// Currently being retried.
    Retrying,
    /// Successfully reprocessed.
    Resolved,
    /// Permanently failed (max retries or manual discard).
    Discarded,
    /// Expired based on age.
    Expired,
}

/// Severity of a dead letter entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

// ── Dead Letter Entry ───────────────────────────────────────────

/// A single dead letter entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeadLetter {
    pub message_id: String,
    pub queue_name: String,
    /// Original message payload.
    pub payload: serde_json::Value,
    /// Original message headers/metadata.
    pub headers: HashMap<String, String>,
    /// Failure reason.
    pub failure_reason: String,
    /// The error type / code.
    pub error_code: Option<String>,
    pub severity: Severity,
    pub state: DlqMessageState,
    pub retry_count: u32,
    pub max_retries: u32,
    /// Original queue / topic the message came from.
    pub original_queue: String,
    pub created_at: DateTime<Utc>,
    pub last_retry_at: Option<DateTime<Utc>>,
    pub resolved_at: Option<DateTime<Utc>>,
    /// Time-to-live in seconds.
    pub ttl_seconds: u64,
}

/// Statistics for a DLQ.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DlqStats {
    pub queue_name: String,
    pub total_messages: u64,
    pub pending: u64,
    pub retrying: u64,
    pub resolved: u64,
    pub discarded: u64,
    pub expired: u64,
    pub avg_retry_count: f64,
    pub oldest_message_age_secs: i64,
}

/// DLQ configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlqConfig {
    pub queue_name: String,
    pub default_max_retries: u32,
    pub default_ttl_seconds: u64,
    pub auto_expire: bool,
}

// ── DLQ Manager ─────────────────────────────────────────────────

/// Manages one or more dead letter queues.
#[derive(Debug, Clone)]
pub struct DlqManager {
    configs: HashMap<String, DlqConfig>,
    messages: Vec<DeadLetter>,
    message_index: HashMap<String, usize>,
    total_energy_uj: u64,
}

impl DlqManager {
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            messages: Vec::new(),
            message_index: HashMap::new(),
            total_energy_uj: 0,
        }
    }

    /// Register a dead letter queue.
    pub fn register_queue(&mut self, config: DlqConfig) {
        self.configs.insert(config.queue_name.clone(), config);
        self.total_energy_uj += 5;
    }

    /// Send a failed message to the DLQ.
    pub fn enqueue(
        &mut self,
        message_id: &str,
        queue_name: &str,
        payload: serde_json::Value,
        headers: HashMap<String, String>,
        failure_reason: &str,
        error_code: Option<&str>,
        severity: Severity,
        original_queue: &str,
    ) -> Result<(), DlqError> {
        let config = self
            .configs
            .get(queue_name)
            .ok_or_else(|| DlqError::QueueNotFound(queue_name.to_string()))?;

        if self.message_index.contains_key(message_id) {
            return Err(DlqError::DuplicateMessage(message_id.to_string()));
        }

        let entry = DeadLetter {
            message_id: message_id.to_string(),
            queue_name: queue_name.to_string(),
            payload,
            headers,
            failure_reason: failure_reason.to_string(),
            error_code: error_code.map(|s| s.to_string()),
            severity,
            state: DlqMessageState::Pending,
            retry_count: 0,
            max_retries: config.default_max_retries,
            original_queue: original_queue.to_string(),
            created_at: Utc::now(),
            last_retry_at: None,
            resolved_at: None,
            ttl_seconds: config.default_ttl_seconds,
        };

        let idx = self.messages.len();
        self.message_index.insert(message_id.to_string(), idx);
        self.messages.push(entry);
        self.total_energy_uj += 8;
        Ok(())
    }

    /// Attempt to retry a message from the DLQ.
    pub fn retry(&mut self, message_id: &str) -> Result<&DeadLetter, DlqError> {
        let idx = *self
            .message_index
            .get(message_id)
            .ok_or_else(|| DlqError::MessageNotFound(message_id.to_string()))?;

        let msg = &self.messages[idx];
        if msg.state == DlqMessageState::Expired {
            return Err(DlqError::MessageExpired(message_id.to_string()));
        }
        if msg.state == DlqMessageState::Discarded || msg.state == DlqMessageState::Resolved {
            return Err(DlqError::MessageNotFound(message_id.to_string()));
        }
        if msg.retry_count >= msg.max_retries {
            return Err(DlqError::MaxRetriesExceeded {
                message_id: message_id.to_string(),
                max_retries: msg.max_retries,
            });
        }

        let msg = &mut self.messages[idx];
        msg.retry_count += 1;
        msg.last_retry_at = Some(Utc::now());
        msg.state = DlqMessageState::Retrying;
        self.total_energy_uj += 10;
        Ok(&self.messages[idx])
    }

    /// Mark a retried message as successfully resolved.
    pub fn resolve(&mut self, message_id: &str) -> Result<(), DlqError> {
        let idx = *self
            .message_index
            .get(message_id)
            .ok_or_else(|| DlqError::MessageNotFound(message_id.to_string()))?;

        let msg = &mut self.messages[idx];
        msg.state = DlqMessageState::Resolved;
        msg.resolved_at = Some(Utc::now());
        self.total_energy_uj += 5;
        Ok(())
    }

    /// Permanently discard a message.
    pub fn discard(&mut self, message_id: &str) -> Result<(), DlqError> {
        let idx = *self
            .message_index
            .get(message_id)
            .ok_or_else(|| DlqError::MessageNotFound(message_id.to_string()))?;

        self.messages[idx].state = DlqMessageState::Discarded;
        self.total_energy_uj += 3;
        Ok(())
    }

    /// Expire messages older than their TTL.
    pub fn expire_old_messages(&mut self) -> u64 {
        let now = Utc::now();
        let mut expired_count = 0u64;

        for msg in &mut self.messages {
            if msg.state == DlqMessageState::Pending || msg.state == DlqMessageState::Retrying {
                let age = now.signed_duration_since(msg.created_at);
                if age > Duration::seconds(msg.ttl_seconds as i64) {
                    msg.state = DlqMessageState::Expired;
                    expired_count += 1;
                }
            }
        }
        self.total_energy_uj += expired_count * 2;
        expired_count
    }

    /// Get a message by id.
    pub fn get_message(&self, message_id: &str) -> Option<&DeadLetter> {
        self.message_index.get(message_id).map(|idx| &self.messages[*idx])
    }

    /// List pending messages for a queue.
    pub fn pending_messages(&self, queue_name: &str) -> Vec<&DeadLetter> {
        self.messages
            .iter()
            .filter(|m| m.queue_name == queue_name && m.state == DlqMessageState::Pending)
            .collect()
    }

    /// Get messages by severity.
    pub fn by_severity(&self, severity: &Severity) -> Vec<&DeadLetter> {
        self.messages
            .iter()
            .filter(|m| &m.severity == severity)
            .collect()
    }

    /// Peek at the oldest pending message in a queue.
    pub fn peek(&self, queue_name: &str) -> Result<&DeadLetter, DlqError> {
        self.messages
            .iter()
            .filter(|m| m.queue_name == queue_name && m.state == DlqMessageState::Pending)
            .min_by_key(|m| m.created_at)
            .ok_or_else(|| DlqError::QueueEmpty(queue_name.to_string()))
    }

    /// Compute stats for a queue.
    pub fn stats(&self, queue_name: &str) -> Result<DlqStats, DlqError> {
        if !self.configs.contains_key(queue_name) {
            return Err(DlqError::QueueNotFound(queue_name.to_string()));
        }

        let msgs: Vec<&DeadLetter> = self
            .messages
            .iter()
            .filter(|m| m.queue_name == queue_name)
            .collect();

        let total = msgs.len() as u64;
        let pending = msgs.iter().filter(|m| m.state == DlqMessageState::Pending).count() as u64;
        let retrying = msgs.iter().filter(|m| m.state == DlqMessageState::Retrying).count() as u64;
        let resolved = msgs.iter().filter(|m| m.state == DlqMessageState::Resolved).count() as u64;
        let discarded =
            msgs.iter().filter(|m| m.state == DlqMessageState::Discarded).count() as u64;
        let expired = msgs.iter().filter(|m| m.state == DlqMessageState::Expired).count() as u64;

        let total_retries: u32 = msgs.iter().map(|m| m.retry_count).sum();
        let avg_retry_count = if total > 0 {
            total_retries as f64 / total as f64
        } else {
            0.0
        };

        let now = Utc::now();
        let oldest_age = msgs
            .iter()
            .map(|m| now.signed_duration_since(m.created_at).num_seconds())
            .max()
            .unwrap_or(0);

        Ok(DlqStats {
            queue_name: queue_name.to_string(),
            total_messages: total,
            pending,
            retrying,
            resolved,
            discarded,
            expired,
            avg_retry_count,
            oldest_message_age_secs: oldest_age,
        })
    }

    /// Total energy consumed.
    pub fn total_energy_uj(&self) -> u64 {
        self.total_energy_uj
    }

    /// Count of all messages across all queues.
    pub fn total_message_count(&self) -> usize {
        self.messages.len()
    }
}

impl Default for DlqManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> DlqManager {
        let mut mgr = DlqManager::new();
        mgr.register_queue(DlqConfig {
            queue_name: "dlq-main".into(),
            default_max_retries: 3,
            default_ttl_seconds: 3600,
            auto_expire: true,
        });
        mgr
    }

    fn enqueue_msg(mgr: &mut DlqManager, id: &str) {
        mgr.enqueue(
            id,
            "dlq-main",
            serde_json::json!({"data": "test"}),
            HashMap::new(),
            "processing failed",
            Some("ERR_500"),
            Severity::Medium,
            "orders-queue",
        )
        .unwrap();
    }

    #[test]
    fn test_register_and_enqueue() {
        let mut mgr = setup();
        enqueue_msg(&mut mgr, "m1");
        assert_eq!(mgr.total_message_count(), 1);
        let msg = mgr.get_message("m1").unwrap();
        assert_eq!(msg.state, DlqMessageState::Pending);
        assert_eq!(msg.retry_count, 0);
        assert_eq!(msg.failure_reason, "processing failed");
        assert_eq!(msg.error_code, Some("ERR_500".into()));
    }

    #[test]
    fn test_duplicate_message() {
        let mut mgr = setup();
        enqueue_msg(&mut mgr, "m1");
        let result = mgr.enqueue(
            "m1",
            "dlq-main",
            serde_json::json!(null),
            HashMap::new(),
            "dup",
            None,
            Severity::Low,
            "q",
        );
        assert_eq!(result, Err(DlqError::DuplicateMessage("m1".into())));
    }

    #[test]
    fn test_queue_not_found() {
        let mut mgr = DlqManager::new();
        let result = mgr.enqueue(
            "m1",
            "nonexistent",
            serde_json::json!(null),
            HashMap::new(),
            "fail",
            None,
            Severity::Low,
            "q",
        );
        assert_eq!(result, Err(DlqError::QueueNotFound("nonexistent".into())));
    }

    #[test]
    fn test_retry_success_cycle() {
        let mut mgr = setup();
        enqueue_msg(&mut mgr, "m1");

        // Retry 1.
        let msg = mgr.retry("m1").unwrap();
        assert_eq!(msg.retry_count, 1);
        assert_eq!(msg.state, DlqMessageState::Retrying);

        // Mark resolved.
        mgr.resolve("m1").unwrap();
        let msg = mgr.get_message("m1").unwrap();
        assert_eq!(msg.state, DlqMessageState::Resolved);
        assert!(msg.resolved_at.is_some());
    }

    #[test]
    fn test_max_retries_exceeded() {
        let mut mgr = setup();
        enqueue_msg(&mut mgr, "m1");

        // Use up all 3 retries.
        for _ in 0..3 {
            mgr.retry("m1").unwrap();
            // Reset to Pending so we can retry again.
            let idx = mgr.message_index["m1"];
            mgr.messages[idx].state = DlqMessageState::Pending;
        }

        // 4th retry should fail.
        assert!(matches!(
            mgr.retry("m1"),
            Err(DlqError::MaxRetriesExceeded { max_retries: 3, .. })
        ));
    }

    #[test]
    fn test_discard() {
        let mut mgr = setup();
        enqueue_msg(&mut mgr, "m1");

        mgr.discard("m1").unwrap();
        let msg = mgr.get_message("m1").unwrap();
        assert_eq!(msg.state, DlqMessageState::Discarded);
    }

    #[test]
    fn test_expire_old_messages() {
        let mut mgr = setup();
        enqueue_msg(&mut mgr, "m1");

        // Manually backdate the message.
        let idx = mgr.message_index["m1"];
        mgr.messages[idx].created_at = Utc::now() - Duration::seconds(7200);
        mgr.messages[idx].ttl_seconds = 3600;

        let expired = mgr.expire_old_messages();
        assert_eq!(expired, 1);
        assert_eq!(mgr.get_message("m1").unwrap().state, DlqMessageState::Expired);
    }

    #[test]
    fn test_expire_does_not_affect_resolved() {
        let mut mgr = setup();
        enqueue_msg(&mut mgr, "m1");
        mgr.resolve("m1").unwrap();

        let idx = mgr.message_index["m1"];
        mgr.messages[idx].created_at = Utc::now() - Duration::seconds(7200);

        let expired = mgr.expire_old_messages();
        assert_eq!(expired, 0);
    }

    #[test]
    fn test_retry_expired_message() {
        let mut mgr = setup();
        enqueue_msg(&mut mgr, "m1");

        let idx = mgr.message_index["m1"];
        mgr.messages[idx].state = DlqMessageState::Expired;

        assert_eq!(mgr.retry("m1"), Err(DlqError::MessageExpired("m1".into())));
    }

    #[test]
    fn test_retry_nonexistent() {
        let mut mgr = setup();
        assert_eq!(
            mgr.retry("missing"),
            Err(DlqError::MessageNotFound("missing".into()))
        );
    }

    #[test]
    fn test_pending_messages() {
        let mut mgr = setup();
        enqueue_msg(&mut mgr, "m1");
        enqueue_msg(&mut mgr, "m2");
        enqueue_msg(&mut mgr, "m3");
        mgr.resolve("m2").unwrap();

        let pending = mgr.pending_messages("dlq-main");
        assert_eq!(pending.len(), 2);
    }

    #[test]
    fn test_by_severity() {
        let mut mgr = setup();
        mgr.enqueue(
            "m1",
            "dlq-main",
            serde_json::json!(null),
            HashMap::new(),
            "low",
            None,
            Severity::Low,
            "q",
        )
        .unwrap();
        mgr.enqueue(
            "m2",
            "dlq-main",
            serde_json::json!(null),
            HashMap::new(),
            "critical",
            None,
            Severity::Critical,
            "q",
        )
        .unwrap();

        assert_eq!(mgr.by_severity(&Severity::Critical).len(), 1);
        assert_eq!(mgr.by_severity(&Severity::Low).len(), 1);
        assert_eq!(mgr.by_severity(&Severity::High).len(), 0);
    }

    #[test]
    fn test_peek() {
        let mut mgr = setup();
        enqueue_msg(&mut mgr, "m1");
        enqueue_msg(&mut mgr, "m2");

        let oldest = mgr.peek("dlq-main").unwrap();
        assert_eq!(oldest.message_id, "m1");
    }

    #[test]
    fn test_peek_empty() {
        let mgr = setup();
        assert_eq!(
            mgr.peek("dlq-main"),
            Err(DlqError::QueueEmpty("dlq-main".into()))
        );
    }

    #[test]
    fn test_stats() {
        let mut mgr = setup();
        enqueue_msg(&mut mgr, "m1");
        enqueue_msg(&mut mgr, "m2");
        enqueue_msg(&mut mgr, "m3");

        mgr.retry("m1").unwrap();
        mgr.resolve("m1").unwrap();
        mgr.discard("m3").unwrap();

        let stats = mgr.stats("dlq-main").unwrap();
        assert_eq!(stats.total_messages, 3);
        assert_eq!(stats.resolved, 1);
        assert_eq!(stats.pending, 1);
        assert_eq!(stats.discarded, 1);
    }

    #[test]
    fn test_stats_queue_not_found() {
        let mgr = DlqManager::new();
        assert_eq!(
            mgr.stats("nope"),
            Err(DlqError::QueueNotFound("nope".into()))
        );
    }

    #[test]
    fn test_energy_tracking() {
        let mut mgr = setup();
        let e1 = mgr.total_energy_uj();
        enqueue_msg(&mut mgr, "m1");
        assert!(mgr.total_energy_uj() > e1);
    }

    #[test]
    fn test_default_manager() {
        let mgr = DlqManager::default();
        assert_eq!(mgr.total_message_count(), 0);
    }

    #[test]
    fn test_error_display() {
        let e = DlqError::MaxRetriesExceeded {
            message_id: "m1".into(),
            max_retries: 5,
        };
        let s = e.to_string();
        assert!(s.contains("5"));
        assert!(s.contains("m1"));
    }

    #[test]
    fn test_message_state_serde() {
        let state = DlqMessageState::Retrying;
        let json = serde_json::to_string(&state).unwrap();
        let parsed: DlqMessageState = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, DlqMessageState::Retrying);
    }

    #[test]
    fn test_severity_serde() {
        let sev = Severity::Critical;
        let json = serde_json::to_string(&sev).unwrap();
        let parsed: Severity = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, Severity::Critical);
    }

    #[test]
    fn test_headers_preserved() {
        let mut mgr = setup();
        let mut headers = HashMap::new();
        headers.insert("x-trace-id".into(), "abc123".into());
        mgr.enqueue(
            "m1",
            "dlq-main",
            serde_json::json!(null),
            headers,
            "fail",
            None,
            Severity::Low,
            "q",
        )
        .unwrap();

        let msg = mgr.get_message("m1").unwrap();
        assert_eq!(msg.headers.get("x-trace-id"), Some(&"abc123".to_string()));
    }

    #[test]
    fn test_retry_discarded_fails() {
        let mut mgr = setup();
        enqueue_msg(&mut mgr, "m1");
        mgr.discard("m1").unwrap();
        assert!(matches!(
            mgr.retry("m1"),
            Err(DlqError::MessageNotFound(_))
        ));
    }
}
