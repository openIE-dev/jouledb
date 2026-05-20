//! Raft consensus log — log entries with term/index, append with consistency
//! check, commit index, log compaction/truncation, snapshot, log matching
//! property verification, and log statistics.

use std::collections::HashMap;

// ── Log Entry ────────────────────────────────────────────────────────────────

/// A single entry in the Raft log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaftEntry {
    /// Term in which this entry was created.
    pub term: u64,
    /// 1-based index in the log.
    pub index: u64,
    /// Command payload.
    pub command: String,
    /// Entry kind.
    pub kind: EntryKind,
}

/// The kind of log entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// Normal client command.
    Command,
    /// No-op entry (used after leader election).
    Noop,
    /// Configuration change entry.
    ConfigChange,
}

// ── Snapshot ─────────────────────────────────────────────────────────────────

/// A snapshot of compacted log state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    /// Last included index.
    pub last_included_index: u64,
    /// Last included term.
    pub last_included_term: u64,
    /// Serialized state machine data.
    pub data: Vec<u8>,
    /// Configuration at the snapshot point.
    pub config: Vec<String>,
}

// ── Log Statistics ───────────────────────────────────────────────────────────

/// Statistics about the Raft log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogStats {
    /// Total number of entries currently in the log.
    pub entry_count: usize,
    /// First index in the log (may be > 1 after compaction).
    pub first_index: u64,
    /// Last index in the log (0 if empty).
    pub last_index: u64,
    /// Last term in the log (0 if empty).
    pub last_term: u64,
    /// Current commit index.
    pub commit_index: u64,
    /// Number of committed entries.
    pub committed_count: usize,
    /// Number of uncommitted entries.
    pub uncommitted_count: usize,
    /// Total bytes of command payloads.
    pub total_payload_bytes: usize,
    /// Map of term -> number of entries in that term.
    pub entries_per_term: HashMap<u64, usize>,
    /// Whether a snapshot exists.
    pub has_snapshot: bool,
}

// ── Error ────────────────────────────────────────────────────────────────────

/// Error type for Raft log operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaftLogError {
    /// Gap in log indices.
    IndexGap { expected: u64, got: u64 },
    /// Term mismatch at the given index during consistency check.
    TermMismatch { index: u64, expected_term: u64, actual_term: u64 },
    /// Attempted to access an index before the snapshot.
    IndexCompacted { requested: u64, first_available: u64 },
    /// Attempted to commit beyond the last index.
    CommitBeyondEnd { commit: u64, last: u64 },
    /// Log is empty.
    EmptyLog,
    /// Invalid snapshot (last_included_index is 0).
    InvalidSnapshot,
}

// ── Raft Log ─────────────────────────────────────────────────────────────────

/// A Raft consensus log supporting append, truncation, compaction, and snapshot.
#[derive(Debug, Clone)]
pub struct RaftLog {
    /// The log entries. May start at an index > 1 after compaction.
    entries: Vec<RaftEntry>,
    /// The commit index. Entries at or before this index are committed.
    commit_index: u64,
    /// The last applied index (for state machine tracking).
    last_applied: u64,
    /// Snapshot, if any.
    snapshot: Option<Snapshot>,
    /// Offset: the index of the first entry in the entries vec.
    /// After compaction, entries[0].index == offset.
    offset: u64,
}

impl RaftLog {
    /// Create a new, empty Raft log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            snapshot: None,
            offset: 1, // 1-based indexing
        }
    }

    /// Return the first valid index in the log.
    pub fn first_index(&self) -> u64 {
        self.offset
    }

    /// Return the last index in the log, or 0 if empty.
    pub fn last_index(&self) -> u64 {
        if self.entries.is_empty() {
            if let Some(snap) = &self.snapshot {
                return snap.last_included_index;
            }
            return 0;
        }
        self.entries[self.entries.len() - 1].index
    }

    /// Return the last term in the log, or 0 if empty.
    pub fn last_term(&self) -> u64 {
        if self.entries.is_empty() {
            if let Some(snap) = &self.snapshot {
                return snap.last_included_term;
            }
            return 0;
        }
        self.entries[self.entries.len() - 1].term
    }

    /// Return the number of entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the commit index.
    pub fn commit_index(&self) -> u64 {
        self.commit_index
    }

    /// Get the last applied index.
    pub fn last_applied(&self) -> u64 {
        self.last_applied
    }

    /// Look up an entry by index.
    pub fn get(&self, index: u64) -> Result<&RaftEntry, RaftLogError> {
        if index < self.offset {
            return Err(RaftLogError::IndexCompacted {
                requested: index,
                first_available: self.offset,
            });
        }
        let pos = (index - self.offset) as usize;
        self.entries.get(pos).ok_or(RaftLogError::IndexGap {
            expected: index,
            got: self.last_index(),
        })
    }

    /// Get the term at a given index. Returns 0 for index 0.
    pub fn term_at(&self, index: u64) -> Result<u64, RaftLogError> {
        if index == 0 {
            return Ok(0);
        }
        // Check snapshot boundary.
        if let Some(snap) = &self.snapshot {
            if index == snap.last_included_index {
                return Ok(snap.last_included_term);
            }
        }
        Ok(self.get(index)?.term)
    }

    /// Append a single entry, checking that the index is contiguous.
    pub fn append(&mut self, entry: RaftEntry) -> Result<(), RaftLogError> {
        let expected = self.last_index() + 1;
        if entry.index != expected {
            return Err(RaftLogError::IndexGap {
                expected,
                got: entry.index,
            });
        }
        self.entries.push(entry);
        Ok(())
    }

    /// Append multiple entries with consistency check. `prev_log_index` and
    /// `prev_log_term` are checked against the existing log (Raft AppendEntries
    /// semantics). On term conflict, the log is truncated from the conflict point.
    pub fn append_entries(
        &mut self,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<RaftEntry>,
    ) -> Result<(), RaftLogError> {
        // Verify the previous entry matches.
        if prev_log_index > 0 {
            let actual_term = self.term_at(prev_log_index)?;
            if actual_term != prev_log_term {
                return Err(RaftLogError::TermMismatch {
                    index: prev_log_index,
                    expected_term: prev_log_term,
                    actual_term,
                });
            }
        }

        // Append entries, handling conflicts.
        for entry in entries {
            let idx = entry.index;
            if idx <= self.last_index() {
                // Entry already exists — check for conflict.
                let existing_term = self.term_at(idx)?;
                if existing_term != entry.term {
                    // Conflict: truncate from this point onward.
                    self.truncate_from(idx);
                    self.entries.push(entry);
                }
                // If terms match, skip (idempotent).
            } else {
                // New entry.
                self.entries.push(entry);
            }
        }
        Ok(())
    }

    /// Truncate all entries from `from_index` onward (inclusive).
    pub fn truncate_from(&mut self, from_index: u64) {
        if from_index < self.offset {
            return;
        }
        let pos = (from_index - self.offset) as usize;
        self.entries.truncate(pos);
        // Adjust commit index if needed.
        if self.commit_index >= from_index {
            self.commit_index = from_index.saturating_sub(1);
        }
        if self.last_applied >= from_index {
            self.last_applied = from_index.saturating_sub(1);
        }
    }

    /// Advance the commit index to `index`. The index must not exceed the last
    /// log index.
    pub fn commit_to(&mut self, index: u64) -> Result<(), RaftLogError> {
        if index > self.last_index() {
            return Err(RaftLogError::CommitBeyondEnd {
                commit: index,
                last: self.last_index(),
            });
        }
        if index > self.commit_index {
            self.commit_index = index;
        }
        Ok(())
    }

    /// Mark entries as applied up to `index`.
    pub fn apply_to(&mut self, index: u64) {
        if index > self.last_applied && index <= self.commit_index {
            self.last_applied = index;
        }
    }

    /// Return all committed but not yet applied entries.
    pub fn unapplied_entries(&self) -> Vec<&RaftEntry> {
        if self.last_applied >= self.commit_index {
            return Vec::new();
        }
        let start = self.last_applied + 1;
        let end = self.commit_index;
        (start..=end)
            .filter_map(|i| self.get(i).ok())
            .collect()
    }

    /// Compact the log up to `up_to_index` (inclusive), replacing compacted
    /// entries with a snapshot.
    pub fn compact(&mut self, up_to_index: u64, state_data: Vec<u8>, config: Vec<String>) -> Result<(), RaftLogError> {
        if up_to_index == 0 {
            return Err(RaftLogError::InvalidSnapshot);
        }
        let term = self.term_at(up_to_index)?;
        self.snapshot = Some(Snapshot {
            last_included_index: up_to_index,
            last_included_term: term,
            data: state_data,
            config,
        });
        // Remove compacted entries.
        let new_start = (up_to_index + 1 - self.offset) as usize;
        if new_start <= self.entries.len() {
            self.entries = self.entries[new_start..].to_vec();
        } else {
            self.entries.clear();
        }
        self.offset = up_to_index + 1;
        Ok(())
    }

    /// Install a snapshot from a leader. Discards all entries covered by the
    /// snapshot.
    pub fn install_snapshot(&mut self, snapshot: Snapshot) {
        let last_idx = snapshot.last_included_index;
        // Discard any entries at or before the snapshot.
        let new_entries: Vec<RaftEntry> = self.entries.iter()
            .filter(|e| e.index > last_idx)
            .cloned()
            .collect();
        self.entries = new_entries;
        self.offset = last_idx + 1;
        if self.commit_index < last_idx {
            self.commit_index = last_idx;
        }
        if self.last_applied < last_idx {
            self.last_applied = last_idx;
        }
        self.snapshot = Some(snapshot);
    }

    /// Get the current snapshot, if any.
    pub fn snapshot(&self) -> Option<&Snapshot> {
        self.snapshot.as_ref()
    }

    /// Verify the log matching property: for any two entries with the same index
    /// and term, they (and all preceding entries) must be identical.
    /// Returns true if the log is internally consistent.
    pub fn verify_log_matching(&self) -> bool {
        for i in 1..self.entries.len() {
            let prev = &self.entries[i - 1];
            let curr = &self.entries[i];
            // Indices must be contiguous.
            if curr.index != prev.index + 1 {
                return false;
            }
            // Terms must be non-decreasing.
            if curr.term < prev.term {
                return false;
            }
        }
        // First entry index must match offset.
        if let Some(first) = self.entries.first() {
            if first.index != self.offset {
                return false;
            }
        }
        true
    }

    /// Check if this log is at least as up-to-date as (last_log_term, last_log_index).
    /// Used for Raft leader election restriction.
    pub fn is_up_to_date(&self, candidate_last_term: u64, candidate_last_index: u64) -> bool {
        let my_last_term = self.last_term();
        let my_last_index = self.last_index();
        if my_last_term != candidate_last_term {
            return my_last_term <= candidate_last_term;
        }
        my_last_index <= candidate_last_index
    }

    /// Get entries in a given index range (inclusive).
    pub fn entries_in_range(&self, start: u64, end: u64) -> Vec<&RaftEntry> {
        (start..=end)
            .filter_map(|i| self.get(i).ok())
            .collect()
    }

    /// Get all entries for a given term.
    pub fn entries_for_term(&self, term: u64) -> Vec<&RaftEntry> {
        self.entries.iter().filter(|e| e.term == term).collect()
    }

    /// Compute log statistics.
    pub fn stats(&self) -> LogStats {
        let mut entries_per_term: HashMap<u64, usize> = HashMap::new();
        let mut total_payload_bytes = 0usize;
        for entry in &self.entries {
            *entries_per_term.entry(entry.term).or_insert(0) += 1;
            total_payload_bytes += entry.command.len();
        }
        let committed_count = if self.commit_index >= self.offset {
            let commit_pos = (self.commit_index - self.offset + 1) as usize;
            commit_pos.min(self.entries.len())
        } else {
            0
        };
        let uncommitted_count = self.entries.len().saturating_sub(committed_count);

        LogStats {
            entry_count: self.entries.len(),
            first_index: self.first_index(),
            last_index: self.last_index(),
            last_term: self.last_term(),
            commit_index: self.commit_index,
            committed_count,
            uncommitted_count,
            total_payload_bytes,
            entries_per_term,
            has_snapshot: self.snapshot.is_some(),
        }
    }

    /// Find the most recent entry index where the term matches `term`,
    /// searching backwards from the end.
    pub fn find_conflict_index(&self, term: u64) -> Option<u64> {
        for entry in self.entries.iter().rev() {
            if entry.term <= term {
                return Some(entry.index);
            }
        }
        None
    }

    /// Create entries helper: builds a RaftEntry from term and command.
    pub fn make_entry(&self, term: u64, command: &str, kind: EntryKind) -> RaftEntry {
        RaftEntry {
            term,
            index: self.last_index() + 1,
            command: command.to_string(),
            kind,
        }
    }
}

impl Default for RaftLog {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(term: u64, index: u64, cmd: &str) -> RaftEntry {
        RaftEntry {
            term,
            index,
            command: cmd.to_string(),
            kind: EntryKind::Command,
        }
    }

    #[test]
    fn new_log_is_empty() {
        let log = RaftLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert_eq!(log.last_index(), 0);
        assert_eq!(log.last_term(), 0);
        assert_eq!(log.commit_index(), 0);
    }

    #[test]
    fn append_single_entry() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "set x=1")).unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log.last_index(), 1);
        assert_eq!(log.last_term(), 1);
    }

    #[test]
    fn append_multiple_entries() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        log.append(entry(1, 2, "b")).unwrap();
        log.append(entry(2, 3, "c")).unwrap();
        assert_eq!(log.len(), 3);
        assert_eq!(log.last_index(), 3);
        assert_eq!(log.last_term(), 2);
    }

    #[test]
    fn append_gap_returns_error() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        let result = log.append(entry(1, 3, "c"));
        assert!(matches!(result, Err(RaftLogError::IndexGap { expected: 2, got: 3 })));
    }

    #[test]
    fn get_entry_by_index() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "first")).unwrap();
        log.append(entry(2, 2, "second")).unwrap();
        let e = log.get(1).unwrap();
        assert_eq!(e.command, "first");
        let e2 = log.get(2).unwrap();
        assert_eq!(e2.term, 2);
    }

    #[test]
    fn term_at_valid_index() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        log.append(entry(3, 2, "b")).unwrap();
        assert_eq!(log.term_at(0).unwrap(), 0);
        assert_eq!(log.term_at(1).unwrap(), 1);
        assert_eq!(log.term_at(2).unwrap(), 3);
    }

    #[test]
    fn append_entries_with_consistency_check() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        log.append(entry(1, 2, "b")).unwrap();
        // Append entries after index 2, term 1.
        let new_entries = vec![entry(2, 3, "c"), entry(2, 4, "d")];
        log.append_entries(2, 1, new_entries).unwrap();
        assert_eq!(log.len(), 4);
        assert_eq!(log.last_index(), 4);
    }

    #[test]
    fn append_entries_term_mismatch() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        let result = log.append_entries(1, 99, vec![entry(2, 2, "b")]);
        assert!(matches!(result, Err(RaftLogError::TermMismatch { .. })));
    }

    #[test]
    fn append_entries_truncates_on_conflict() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        log.append(entry(1, 2, "b")).unwrap();
        log.append(entry(1, 3, "old")).unwrap();
        // New leader sends entry at index 3 with term 2.
        log.append_entries(2, 1, vec![entry(2, 3, "new")]).unwrap();
        assert_eq!(log.len(), 3);
        assert_eq!(log.get(3).unwrap().term, 2);
        assert_eq!(log.get(3).unwrap().command, "new");
    }

    #[test]
    fn commit_and_apply() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        log.append(entry(1, 2, "b")).unwrap();
        log.append(entry(1, 3, "c")).unwrap();
        log.commit_to(2).unwrap();
        assert_eq!(log.commit_index(), 2);
        let unapplied = log.unapplied_entries();
        assert_eq!(unapplied.len(), 2);
        log.apply_to(1);
        assert_eq!(log.last_applied(), 1);
        let unapplied2 = log.unapplied_entries();
        assert_eq!(unapplied2.len(), 1);
    }

    #[test]
    fn commit_beyond_end_fails() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        let result = log.commit_to(5);
        assert!(matches!(result, Err(RaftLogError::CommitBeyondEnd { .. })));
    }

    #[test]
    fn truncate_from() {
        let mut log = RaftLog::new();
        for i in 1..=5 {
            log.append(entry(1, i, &format!("cmd{}", i))).unwrap();
        }
        log.commit_to(4).unwrap();
        log.truncate_from(3);
        assert_eq!(log.len(), 2);
        assert_eq!(log.last_index(), 2);
        // Commit index should be adjusted.
        assert_eq!(log.commit_index(), 2);
    }

    #[test]
    fn compact_log_creates_snapshot() {
        let mut log = RaftLog::new();
        for i in 1..=5 {
            log.append(entry(1, i, &format!("cmd{}", i))).unwrap();
        }
        log.commit_to(5).unwrap();
        log.compact(3, b"state".to_vec(), vec!["node1".into()]).unwrap();
        assert!(log.snapshot().is_some());
        let snap = log.snapshot().unwrap();
        assert_eq!(snap.last_included_index, 3);
        assert_eq!(snap.last_included_term, 1);
        assert_eq!(log.len(), 2); // entries 4 and 5 remain
        assert_eq!(log.first_index(), 4);
    }

    #[test]
    fn compacted_index_returns_error() {
        let mut log = RaftLog::new();
        for i in 1..=5 {
            log.append(entry(1, i, &format!("cmd{}", i))).unwrap();
        }
        log.commit_to(5).unwrap();
        log.compact(3, b"state".to_vec(), vec![]).unwrap();
        let result = log.get(1);
        assert!(matches!(result, Err(RaftLogError::IndexCompacted { .. })));
    }

    #[test]
    fn install_snapshot() {
        let mut log = RaftLog::new();
        for i in 1..=3 {
            log.append(entry(1, i, &format!("cmd{}", i))).unwrap();
        }
        let snap = Snapshot {
            last_included_index: 10,
            last_included_term: 3,
            data: b"full_state".to_vec(),
            config: vec!["n1".into(), "n2".into()],
        };
        log.install_snapshot(snap);
        assert_eq!(log.commit_index(), 10);
        assert_eq!(log.last_applied(), 10);
        assert!(log.is_empty()); // all old entries were before index 10
        assert_eq!(log.last_index(), 10); // from snapshot
        assert_eq!(log.last_term(), 3);
    }

    #[test]
    fn verify_log_matching_valid() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        log.append(entry(1, 2, "b")).unwrap();
        log.append(entry(2, 3, "c")).unwrap();
        assert!(log.verify_log_matching());
    }

    #[test]
    fn is_up_to_date_same_term() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        log.append(entry(1, 2, "b")).unwrap();
        // Candidate has same term but longer log.
        assert!(log.is_up_to_date(1, 3));
        // Candidate has same term, same length.
        assert!(log.is_up_to_date(1, 2));
        // Candidate has same term but shorter log.
        assert!(!log.is_up_to_date(1, 1));
    }

    #[test]
    fn is_up_to_date_different_term() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        // Candidate has higher term — always more up-to-date.
        assert!(log.is_up_to_date(2, 1));
        // Candidate has lower term — never more up-to-date.
        assert!(!log.is_up_to_date(0, 5));
    }

    #[test]
    fn entries_for_term() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        log.append(entry(1, 2, "b")).unwrap();
        log.append(entry(2, 3, "c")).unwrap();
        log.append(entry(2, 4, "d")).unwrap();
        log.append(entry(3, 5, "e")).unwrap();
        let t1 = log.entries_for_term(1);
        assert_eq!(t1.len(), 2);
        let t2 = log.entries_for_term(2);
        assert_eq!(t2.len(), 2);
        let t3 = log.entries_for_term(3);
        assert_eq!(t3.len(), 1);
    }

    #[test]
    fn entries_in_range() {
        let mut log = RaftLog::new();
        for i in 1..=5 {
            log.append(entry(1, i, &format!("cmd{}", i))).unwrap();
        }
        let range = log.entries_in_range(2, 4);
        assert_eq!(range.len(), 3);
        assert_eq!(range[0].index, 2);
        assert_eq!(range[2].index, 4);
    }

    #[test]
    fn stats() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "abc")).unwrap();
        log.append(entry(1, 2, "de")).unwrap();
        log.append(entry(2, 3, "f")).unwrap();
        log.commit_to(2).unwrap();
        let s = log.stats();
        assert_eq!(s.entry_count, 3);
        assert_eq!(s.first_index, 1);
        assert_eq!(s.last_index, 3);
        assert_eq!(s.last_term, 2);
        assert_eq!(s.commit_index, 2);
        assert_eq!(s.committed_count, 2);
        assert_eq!(s.uncommitted_count, 1);
        assert_eq!(s.total_payload_bytes, 6); // "abc" + "de" + "f"
        assert!(!s.has_snapshot);
    }

    #[test]
    fn find_conflict_index() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        log.append(entry(2, 2, "b")).unwrap();
        log.append(entry(3, 3, "c")).unwrap();
        assert_eq!(log.find_conflict_index(2), Some(2));
        assert_eq!(log.find_conflict_index(1), Some(1));
        assert_eq!(log.find_conflict_index(5), Some(3));
        assert_eq!(log.find_conflict_index(0), None);
    }

    #[test]
    fn make_entry_helper() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        let e = log.make_entry(2, "b", EntryKind::Noop);
        assert_eq!(e.index, 2);
        assert_eq!(e.term, 2);
        assert_eq!(e.kind, EntryKind::Noop);
    }

    #[test]
    fn snapshot_term_at_boundary() {
        let mut log = RaftLog::new();
        for i in 1..=5 {
            log.append(entry(i, i, &format!("cmd{}", i))).unwrap();
        }
        log.commit_to(5).unwrap();
        log.compact(3, b"snap".to_vec(), vec![]).unwrap();
        // term_at snapshot boundary returns snapshot term.
        assert_eq!(log.term_at(3).unwrap(), 3);
        // term_at post-snapshot entries works.
        assert_eq!(log.term_at(4).unwrap(), 4);
    }

    #[test]
    fn default_trait() {
        let log = RaftLog::default();
        assert!(log.is_empty());
    }

    #[test]
    fn idempotent_append_entries() {
        let mut log = RaftLog::new();
        log.append(entry(1, 1, "a")).unwrap();
        log.append(entry(1, 2, "b")).unwrap();
        // Re-sending the same entries is idempotent.
        log.append_entries(0, 0, vec![entry(1, 1, "a"), entry(1, 2, "b")]).unwrap();
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn compact_invalid_snapshot() {
        let mut log = RaftLog::new();
        let result = log.compact(0, vec![], vec![]);
        assert!(matches!(result, Err(RaftLogError::InvalidSnapshot)));
    }
}
