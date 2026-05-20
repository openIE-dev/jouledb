//! Transactional outbox pattern — reliable event publishing, idempotent
//! processing, ordered delivery per aggregate, polling publisher, message
//! relay, and cleanup of processed entries.
//!
//! Replaces transactional outbox implementations (Debezium, MassTransit
//! outbox) with a pure-Rust outbox that simulates the outbox table pattern
//! for reliable, exactly-once event delivery across service boundaries.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Errors ─────────────────────────────────────────────────────

/// Outbox domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutboxError {
    /// Duplicate entry (idempotent rejection).
    DuplicateEntry(String),
    /// Entry not found.
    EntryNotFound(String),
    /// Already published.
    AlreadyPublished(String),
    /// Already processed.
    AlreadyProcessed(String),
    /// Relay failed.
    RelayFailed { entry_id: String, reason: String },
    /// Outbox is locked (processing in progress).
    Locked,
}

impl std::fmt::Display for OutboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateEntry(id) => write!(f, "duplicate outbox entry: {id}"),
            Self::EntryNotFound(id) => write!(f, "outbox entry not found: {id}"),
            Self::AlreadyPublished(id) => write!(f, "already published: {id}"),
            Self::AlreadyProcessed(id) => write!(f, "already processed: {id}"),
            Self::RelayFailed { entry_id, reason } => {
                write!(f, "relay failed for {entry_id}: {reason}")
            }
            Self::Locked => write!(f, "outbox is locked"),
        }
    }
}

impl std::error::Error for OutboxError {}

// ── Entry Status ──────────────────────────────────────────────

/// Lifecycle status of an outbox entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntryStatus {
    /// Written to outbox, not yet published.
    Pending,
    /// Published (relayed) to the message broker.
    Published,
    /// Acknowledged by the consumer (processed).
    Processed,
    /// Relay failed.
    Failed,
}

// ── Outbox Entry ──────────────────────────────────────────────

/// An entry in the transactional outbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutboxEntry {
    pub id: String,
    pub aggregate_id: String,
    pub aggregate_type: String,
    pub event_type: String,
    pub payload: String,
    pub status: EntryStatus,
    pub created_at_ms: u64,
    pub published_at_ms: Option<u64>,
    pub processed_at_ms: Option<u64>,
    pub attempt: u32,
    pub max_attempts: u32,
    pub error: Option<String>,
    pub sequence: u64,
    pub headers: HashMap<String, String>,
}

impl OutboxEntry {
    pub fn new(
        id: impl Into<String>,
        aggregate_id: impl Into<String>,
        aggregate_type: impl Into<String>,
        event_type: impl Into<String>,
        payload: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            aggregate_id: aggregate_id.into(),
            aggregate_type: aggregate_type.into(),
            event_type: event_type.into(),
            payload: payload.into(),
            status: EntryStatus::Pending,
            created_at_ms: 0,
            published_at_ms: None,
            processed_at_ms: None,
            attempt: 0,
            max_attempts: 3,
            error: None,
            sequence: 0,
            headers: HashMap::new(),
        }
    }

    pub fn with_max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = n;
        self
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

// ── Relay Result ──────────────────────────────────────────────

/// Result of relaying a batch of outbox entries.
#[derive(Debug, Clone)]
pub struct RelayResult {
    pub published: Vec<String>,
    pub failed: Vec<(String, String)>,
    pub total_attempted: usize,
}

// ── Outbox Stats ──────────────────────────────────────────────

/// Statistics for the outbox.
#[derive(Debug, Clone, Default)]
pub struct OutboxStats {
    pub total_entries: u64,
    pub total_published: u64,
    pub total_processed: u64,
    pub total_failed: u64,
    pub pending_count: usize,
    pub cleaned_up: u64,
}

// ── Processor State ───────────────────────────────────────────

/// Tracks idempotent processing state.
#[derive(Debug, Clone)]
struct ProcessorState {
    /// Set of processed entry IDs (for deduplication).
    processed_ids: HashSet<String>,
}

impl ProcessorState {
    fn new() -> Self {
        Self {
            processed_ids: HashSet::new(),
        }
    }
}

// ── Transactional Outbox ──────────────────────────────────────

/// The transactional outbox.
#[derive(Debug)]
pub struct Outbox {
    /// All entries in insertion order.
    entries: VecDeque<OutboxEntry>,
    /// Index by ID for fast lookup.
    index: HashMap<String, usize>,
    /// Per-aggregate sequence counters.
    aggregate_sequences: HashMap<String, u64>,
    /// Processor state for idempotency.
    processor: ProcessorState,
    /// Simulated clock.
    clock_ms: u64,
    /// Whether a relay operation is in progress.
    locked: bool,
    /// Stats.
    stats: OutboxStats,
    /// IDs that should fail relay (for testing).
    relay_failures: HashSet<String>,
}

impl Default for Outbox {
    fn default() -> Self {
        Self::new()
    }
}

impl Outbox {
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            index: HashMap::new(),
            aggregate_sequences: HashMap::new(),
            processor: ProcessorState::new(),
            clock_ms: 0,
            locked: false,
            stats: OutboxStats::default(),
            relay_failures: HashSet::new(),
        }
    }

    pub fn advance_time(&mut self, ms: u64) {
        self.clock_ms += ms;
    }

    pub fn set_clock(&mut self, ms: u64) {
        self.clock_ms = ms;
    }

    /// Configure an entry ID to fail during relay (for testing).
    pub fn set_relay_failure(&mut self, entry_id: impl Into<String>) {
        self.relay_failures.insert(entry_id.into());
    }

    /// Clear relay failure configuration.
    pub fn clear_relay_failure(&mut self, entry_id: &str) {
        self.relay_failures.remove(entry_id);
    }

    // ── Write ─────────────────────────────────────────────────

    /// Write an entry to the outbox (typically in the same transaction as the
    /// domain state change).
    pub fn write(&mut self, mut entry: OutboxEntry) -> Result<(), OutboxError> {
        if self.index.contains_key(&entry.id) {
            return Err(OutboxError::DuplicateEntry(entry.id));
        }
        entry.created_at_ms = self.clock_ms;
        // Assign sequence within aggregate.
        let seq = self
            .aggregate_sequences
            .entry(entry.aggregate_id.clone())
            .or_insert(0);
        *seq += 1;
        entry.sequence = *seq;

        let idx = self.entries.len();
        self.index.insert(entry.id.clone(), idx);
        self.entries.push_back(entry);
        self.stats.total_entries += 1;
        self.stats.pending_count += 1;
        Ok(())
    }

    // ── Polling Publisher ─────────────────────────────────────

    /// Poll for pending entries (up to `batch_size`).
    pub fn poll_pending(&self, batch_size: usize) -> Vec<&OutboxEntry> {
        self.entries
            .iter()
            .filter(|e| e.status == EntryStatus::Pending || e.status == EntryStatus::Failed)
            .filter(|e| e.attempt < e.max_attempts)
            .take(batch_size)
            .collect()
    }

    /// Poll for pending entries for a specific aggregate, in sequence order.
    pub fn poll_for_aggregate(&self, aggregate_id: &str, batch_size: usize) -> Vec<&OutboxEntry> {
        let mut entries: Vec<&OutboxEntry> = self
            .entries
            .iter()
            .filter(|e| {
                e.aggregate_id == aggregate_id
                    && (e.status == EntryStatus::Pending || e.status == EntryStatus::Failed)
                    && e.attempt < e.max_attempts
            })
            .collect();
        entries.sort_by_key(|e| e.sequence);
        entries.into_iter().take(batch_size).collect()
    }

    // ── Relay (Publish) ──────────────────────────────────────

    /// Relay (publish) a batch of pending entries. Returns relay results.
    pub fn relay(&mut self, batch_size: usize) -> Result<RelayResult, OutboxError> {
        if self.locked {
            return Err(OutboxError::Locked);
        }
        self.locked = true;

        let pending_ids: Vec<String> = self
            .entries
            .iter()
            .filter(|e| e.status == EntryStatus::Pending || e.status == EntryStatus::Failed)
            .filter(|e| e.attempt < e.max_attempts)
            .take(batch_size)
            .map(|e| e.id.clone())
            .collect();

        let mut result = RelayResult {
            published: Vec::new(),
            failed: Vec::new(),
            total_attempted: pending_ids.len(),
        };

        for entry_id in pending_ids {
            let should_fail = self.relay_failures.contains(&entry_id);
            if let Some(idx) = self.index.get(&entry_id).copied() {
                if let Some(entry) = self.entries.get_mut(idx) {
                    entry.attempt += 1;
                    if should_fail {
                        entry.status = EntryStatus::Failed;
                        entry.error = Some("relay failed (simulated)".to_string());
                        result
                            .failed
                            .push((entry_id, "relay failed (simulated)".to_string()));
                        self.stats.total_failed += 1;
                    } else {
                        entry.status = EntryStatus::Published;
                        entry.published_at_ms = Some(self.clock_ms);
                        result.published.push(entry_id);
                        self.stats.total_published += 1;
                        self.stats.pending_count = self.stats.pending_count.saturating_sub(1);
                    }
                }
            }
        }

        self.locked = false;
        Ok(result)
    }

    /// Relay entries for a specific aggregate in order.
    pub fn relay_for_aggregate(
        &mut self,
        aggregate_id: &str,
        batch_size: usize,
    ) -> Result<RelayResult, OutboxError> {
        if self.locked {
            return Err(OutboxError::Locked);
        }
        self.locked = true;

        let mut entries_to_relay: Vec<(String, u64)> = self
            .entries
            .iter()
            .filter(|e| {
                e.aggregate_id == aggregate_id
                    && (e.status == EntryStatus::Pending || e.status == EntryStatus::Failed)
                    && e.attempt < e.max_attempts
            })
            .map(|e| (e.id.clone(), e.sequence))
            .collect();
        entries_to_relay.sort_by_key(|(_, seq)| *seq);
        let entry_ids: Vec<String> = entries_to_relay
            .into_iter()
            .take(batch_size)
            .map(|(id, _)| id)
            .collect();

        let mut result = RelayResult {
            published: Vec::new(),
            failed: Vec::new(),
            total_attempted: entry_ids.len(),
        };

        for entry_id in entry_ids {
            let should_fail = self.relay_failures.contains(&entry_id);
            if let Some(idx) = self.index.get(&entry_id).copied() {
                if let Some(entry) = self.entries.get_mut(idx) {
                    entry.attempt += 1;
                    if should_fail {
                        entry.status = EntryStatus::Failed;
                        entry.error = Some("relay failed (simulated)".to_string());
                        result
                            .failed
                            .push((entry_id, "relay failed (simulated)".to_string()));
                        self.stats.total_failed += 1;
                    } else {
                        entry.status = EntryStatus::Published;
                        entry.published_at_ms = Some(self.clock_ms);
                        result.published.push(entry_id);
                        self.stats.total_published += 1;
                        self.stats.pending_count = self.stats.pending_count.saturating_sub(1);
                    }
                }
            }
        }

        self.locked = false;
        Ok(result)
    }

    // ── Process (Consumer Side) ──────────────────────────────

    /// Mark an entry as processed (idempotent — duplicate calls are ignored).
    pub fn mark_processed(&mut self, entry_id: &str) -> Result<bool, OutboxError> {
        // Idempotency check.
        if self.processor.processed_ids.contains(entry_id) {
            return Ok(false); // Already processed, no-op.
        }

        let idx = self
            .index
            .get(entry_id)
            .copied()
            .ok_or_else(|| OutboxError::EntryNotFound(entry_id.to_string()))?;
        let entry = self.entries.get_mut(idx).unwrap();

        if entry.status == EntryStatus::Processed {
            return Err(OutboxError::AlreadyProcessed(entry_id.to_string()));
        }

        entry.status = EntryStatus::Processed;
        entry.processed_at_ms = Some(self.clock_ms);
        self.processor.processed_ids.insert(entry_id.to_string());
        self.stats.total_processed += 1;
        Ok(true)
    }

    /// Check if an entry has been processed (idempotency check).
    pub fn is_processed(&self, entry_id: &str) -> bool {
        self.processor.processed_ids.contains(entry_id)
    }

    // ── Cleanup ──────────────────────────────────────────────

    /// Clean up processed entries older than `retention_ms`.
    pub fn cleanup(&mut self, retention_ms: u64) -> usize {
        let cutoff = self.clock_ms.saturating_sub(retention_ms);
        let before = self.entries.len();
        let mut removed_ids = Vec::new();
        self.entries.retain(|e| {
            if e.status == EntryStatus::Processed {
                if let Some(processed_at) = e.processed_at_ms {
                    if processed_at <= cutoff {
                        removed_ids.push(e.id.clone());
                        return false;
                    }
                }
            }
            true
        });
        for id in &removed_ids {
            self.index.remove(id);
        }
        // Rebuild index after removal.
        self.rebuild_index();
        let removed = before - self.entries.len();
        self.stats.cleaned_up += removed as u64;
        removed
    }

    fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, entry) in self.entries.iter().enumerate() {
            self.index.insert(entry.id.clone(), i);
        }
    }

    // ── Queries ──────────────────────────────────────────────

    /// Get an entry by ID.
    pub fn get_entry(&self, id: &str) -> Option<&OutboxEntry> {
        self.index.get(id).and_then(|idx| self.entries.get(*idx))
    }

    /// Get all entries for an aggregate, sorted by sequence.
    pub fn entries_for_aggregate(&self, aggregate_id: &str) -> Vec<&OutboxEntry> {
        let mut entries: Vec<&OutboxEntry> = self
            .entries
            .iter()
            .filter(|e| e.aggregate_id == aggregate_id)
            .collect();
        entries.sort_by_key(|e| e.sequence);
        entries
    }

    /// Get entries by status.
    pub fn entries_by_status(&self, status: EntryStatus) -> Vec<&OutboxEntry> {
        self.entries.iter().filter(|e| e.status == status).collect()
    }

    /// Outbox stats.
    pub fn stats(&self) -> &OutboxStats {
        &self.stats
    }

    /// Total entry count.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Pending count.
    pub fn pending_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.status == EntryStatus::Pending)
            .count()
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(id: &str, agg: &str) -> OutboxEntry {
        OutboxEntry::new(id, agg, "Order", "OrderCreated", "payload")
    }

    #[test]
    fn test_write_entry() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        assert_eq!(outbox.entry_count(), 1);
        let entry = outbox.get_entry("e1").unwrap();
        assert_eq!(entry.status, EntryStatus::Pending);
    }

    #[test]
    fn test_duplicate_entry() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        assert!(matches!(
            outbox.write(make_entry("e1", "order-1")),
            Err(OutboxError::DuplicateEntry(_))
        ));
    }

    #[test]
    fn test_sequence_ordering() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        outbox.write(make_entry("e2", "order-1")).unwrap();
        outbox.write(make_entry("e3", "order-1")).unwrap();
        let entries = outbox.entries_for_aggregate("order-1");
        assert_eq!(entries[0].sequence, 1);
        assert_eq!(entries[1].sequence, 2);
        assert_eq!(entries[2].sequence, 3);
    }

    #[test]
    fn test_relay_publishes() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        outbox.write(make_entry("e2", "order-1")).unwrap();
        let result = outbox.relay(10).unwrap();
        assert_eq!(result.published.len(), 2);
        assert_eq!(outbox.get_entry("e1").unwrap().status, EntryStatus::Published);
    }

    #[test]
    fn test_relay_failure() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        outbox.set_relay_failure("e1");
        let result = outbox.relay(10).unwrap();
        assert_eq!(result.failed.len(), 1);
        assert_eq!(outbox.get_entry("e1").unwrap().status, EntryStatus::Failed);
    }

    #[test]
    fn test_relay_retry_on_failure() {
        let mut outbox = Outbox::new();
        outbox
            .write(make_entry("e1", "order-1").with_max_attempts(3))
            .unwrap();
        outbox.set_relay_failure("e1");
        outbox.relay(10).unwrap(); // attempt 1 — fails
        outbox.clear_relay_failure("e1");
        let result = outbox.relay(10).unwrap(); // attempt 2 — succeeds
        assert_eq!(result.published.len(), 1);
        assert_eq!(outbox.get_entry("e1").unwrap().attempt, 2);
    }

    #[test]
    fn test_max_relay_attempts() {
        let mut outbox = Outbox::new();
        outbox
            .write(make_entry("e1", "order-1").with_max_attempts(2))
            .unwrap();
        outbox.set_relay_failure("e1");
        outbox.relay(10).unwrap(); // attempt 1
        outbox.relay(10).unwrap(); // attempt 2
        // Should not be polled again.
        let pending = outbox.poll_pending(10);
        assert!(pending.is_empty());
    }

    #[test]
    fn test_mark_processed_idempotent() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        outbox.relay(10).unwrap();
        let first = outbox.mark_processed("e1").unwrap();
        assert!(first);
        // Second call is idempotent — returns false.
        let second = outbox.mark_processed("e1").unwrap();
        assert!(!second);
    }

    #[test]
    fn test_is_processed() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        assert!(!outbox.is_processed("e1"));
        outbox.relay(10).unwrap();
        outbox.mark_processed("e1").unwrap();
        assert!(outbox.is_processed("e1"));
    }

    #[test]
    fn test_cleanup_processed() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        outbox.relay(10).unwrap();
        outbox.mark_processed("e1").unwrap();
        outbox.advance_time(1000);
        let removed = outbox.cleanup(500); // retain for 500ms, entry processed at 0.
        assert_eq!(removed, 1);
        assert_eq!(outbox.entry_count(), 0);
    }

    #[test]
    fn test_cleanup_retains_recent() {
        let mut outbox = Outbox::new();
        outbox.set_clock(100);
        outbox.write(make_entry("e1", "order-1")).unwrap();
        outbox.relay(10).unwrap();
        outbox.mark_processed("e1").unwrap();
        outbox.set_clock(200);
        let removed = outbox.cleanup(500); // cutoff = 200 - 500, no entries eligible.
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_poll_pending() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        outbox.write(make_entry("e2", "order-1")).unwrap();
        let pending = outbox.poll_pending(10);
        assert_eq!(pending.len(), 2);
    }

    #[test]
    fn test_poll_for_aggregate() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        outbox.write(make_entry("e2", "order-2")).unwrap();
        outbox.write(make_entry("e3", "order-1")).unwrap();
        let entries = outbox.poll_for_aggregate("order-1", 10);
        assert_eq!(entries.len(), 2);
        assert!(entries[0].sequence < entries[1].sequence);
    }

    #[test]
    fn test_relay_for_aggregate() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        outbox.write(make_entry("e2", "order-2")).unwrap();
        outbox.write(make_entry("e3", "order-1")).unwrap();
        let result = outbox.relay_for_aggregate("order-1", 10).unwrap();
        assert_eq!(result.published.len(), 2);
        // order-2 should still be pending.
        assert_eq!(
            outbox.get_entry("e2").unwrap().status,
            EntryStatus::Pending
        );
    }

    #[test]
    fn test_entries_by_status() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        outbox.write(make_entry("e2", "order-1")).unwrap();
        outbox.relay(1).unwrap();
        let published = outbox.entries_by_status(EntryStatus::Published);
        assert_eq!(published.len(), 1);
        let pending = outbox.entries_by_status(EntryStatus::Pending);
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_stats() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        outbox.write(make_entry("e2", "order-1")).unwrap();
        assert_eq!(outbox.stats().total_entries, 2);
        outbox.relay(10).unwrap();
        assert_eq!(outbox.stats().total_published, 2);
        outbox.mark_processed("e1").unwrap();
        assert_eq!(outbox.stats().total_processed, 1);
    }

    #[test]
    fn test_entry_headers() {
        let entry = make_entry("e1", "order-1")
            .with_header("correlation-id", "abc-123")
            .with_header("source", "api");
        assert_eq!(entry.headers.get("correlation-id").unwrap(), "abc-123");
    }

    #[test]
    fn test_entry_not_found() {
        let mut outbox = Outbox::new();
        assert!(matches!(
            outbox.mark_processed("nope"),
            Err(OutboxError::EntryNotFound(_))
        ));
    }

    #[test]
    fn test_separate_aggregate_sequences() {
        let mut outbox = Outbox::new();
        outbox.write(make_entry("e1", "order-1")).unwrap();
        outbox.write(make_entry("e2", "order-2")).unwrap();
        outbox.write(make_entry("e3", "order-1")).unwrap();
        // order-1: sequences 1, 2; order-2: sequence 1.
        assert_eq!(outbox.get_entry("e1").unwrap().sequence, 1);
        assert_eq!(outbox.get_entry("e2").unwrap().sequence, 1);
        assert_eq!(outbox.get_entry("e3").unwrap().sequence, 2);
    }
}
