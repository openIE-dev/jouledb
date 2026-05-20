//! Network conflict resolution — vector clocks, strategies, and three-way merge.
//!
//! Provides `ConflictStrategy` enum (LastWriterWins, FirstWriterWins, Merge, Custom),
//! `ConflictResolver` that detects and resolves concurrent updates using vector
//! clocks or timestamps, merge function for compatible changes, conflict log,
//! automatic vs manual resolution mode, resolution statistics, and three-way
//! merge support.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Conflict resolution domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictError {
    /// Key not found.
    KeyNotFound(String),
    /// Cannot auto-resolve this conflict.
    ManualResolutionRequired { key: String },
    /// Merge function failed.
    MergeFailed { key: String, reason: String },
    /// Invalid vector clock (node not found).
    UnknownNode(u64),
}

impl fmt::Display for ConflictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyNotFound(k) => write!(f, "key not found: {k}"),
            Self::ManualResolutionRequired { key } => {
                write!(f, "manual resolution required for key: {key}")
            }
            Self::MergeFailed { key, reason } => {
                write!(f, "merge failed for key '{key}': {reason}")
            }
            Self::UnknownNode(id) => write!(f, "unknown node: {id}"),
        }
    }
}

impl std::error::Error for ConflictError {}

// ── Vector Clock ────────────────────────────────────────────────

/// A vector clock for tracking causality across nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorClock {
    counters: HashMap<u64, u64>,
}

impl VectorClock {
    pub fn new() -> Self {
        Self { counters: HashMap::new() }
    }

    /// Increment the counter for a node.
    pub fn increment(&mut self, node_id: u64) {
        let c = self.counters.entry(node_id).or_insert(0);
        *c += 1;
    }

    /// Get the counter for a node.
    pub fn get(&self, node_id: u64) -> u64 {
        self.counters.get(&node_id).copied().unwrap_or(0)
    }

    /// Merge another vector clock (take max per node).
    pub fn merge(&mut self, other: &VectorClock) {
        for (&node, &count) in &other.counters {
            let c = self.counters.entry(node).or_insert(0);
            *c = (*c).max(count);
        }
    }

    /// Check if this clock happened-before another.
    pub fn happened_before(&self, other: &VectorClock) -> bool {
        let mut at_least_one_less = false;
        for &node in self.counters.keys().chain(other.counters.keys()).collect::<std::collections::HashSet<_>>().iter() {
            let s = self.get(*node);
            let o = other.get(*node);
            if s > o {
                return false;
            }
            if s < o {
                at_least_one_less = true;
            }
        }
        at_least_one_less
    }

    /// Check if two clocks are concurrent (neither happened-before the other).
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        !self.happened_before(other) && !other.happened_before(self) && self != other
    }

    /// All node IDs in the clock.
    pub fn nodes(&self) -> Vec<u64> {
        self.counters.keys().copied().collect()
    }
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for VectorClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut entries: Vec<_> = self.counters.iter().collect();
        entries.sort_by_key(|(k, _)| *k);
        let parts: Vec<String> = entries.iter().map(|(k, v)| format!("{}:{}", k, v)).collect();
        write!(f, "VClock({})", parts.join(", "))
    }
}

// ── Conflict Strategy ───────────────────────────────────────────

/// Strategy for resolving conflicting concurrent updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictStrategy {
    /// Highest timestamp wins.
    LastWriterWins,
    /// Lowest timestamp wins (first writer preserved).
    FirstWriterWins,
    /// Attempt to merge values (e.g., set union).
    Merge,
    /// Require manual intervention.
    Manual,
}

impl fmt::Display for ConflictStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::LastWriterWins => "last-writer-wins",
            Self::FirstWriterWins => "first-writer-wins",
            Self::Merge => "merge",
            Self::Manual => "manual",
        };
        write!(f, "{s}")
    }
}

// ── Versioned Value ─────────────────────────────────────────────

/// A value with a vector clock and wall-clock timestamp.
#[derive(Debug, Clone)]
pub struct VersionedValue {
    pub data: Vec<u8>,
    pub clock: VectorClock,
    pub timestamp_ms: u64,
    pub writer_node: u64,
}

impl VersionedValue {
    pub fn new(data: Vec<u8>, writer_node: u64, timestamp_ms: u64) -> Self {
        let mut clock = VectorClock::new();
        clock.increment(writer_node);
        Self { data, clock, timestamp_ms, writer_node }
    }
}

// ── Conflict Record ─────────────────────────────────────────────

/// Record of a detected and (optionally) resolved conflict.
#[derive(Debug, Clone)]
pub struct ConflictRecord {
    pub key: String,
    pub local_timestamp: u64,
    pub remote_timestamp: u64,
    pub strategy: ConflictStrategy,
    pub resolved: bool,
    pub winner_node: Option<u64>,
}

impl fmt::Display for ConflictRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Conflict(key='{}', strategy={}, resolved={})",
            self.key, self.strategy, self.resolved,
        )
    }
}

// ── Resolution Statistics ───────────────────────────────────────

/// Conflict resolution statistics.
#[derive(Debug, Clone, Default)]
pub struct ResolutionStats {
    pub conflicts_detected: u64,
    pub auto_resolved: u64,
    pub manual_resolved: u64,
    pub merge_attempts: u64,
    pub merge_successes: u64,
}

impl fmt::Display for ResolutionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Stats(detected={}, auto={}, manual={}, merges={}/{})",
            self.conflicts_detected,
            self.auto_resolved,
            self.manual_resolved,
            self.merge_successes,
            self.merge_attempts,
        )
    }
}

// ── Conflict Resolver ───────────────────────────────────────────

/// Detects and resolves concurrent updates using vector clocks.
pub struct ConflictResolver {
    pub strategy: ConflictStrategy,
    store: HashMap<String, VersionedValue>,
    conflict_log: Vec<ConflictRecord>,
    pub stats: ResolutionStats,
    /// Unresolved conflicts needing manual resolution.
    unresolved: HashMap<String, (VersionedValue, VersionedValue)>,
}

impl ConflictResolver {
    pub fn new(strategy: ConflictStrategy) -> Self {
        Self {
            strategy,
            store: HashMap::new(),
            conflict_log: Vec::new(),
            stats: ResolutionStats::default(),
            unresolved: HashMap::new(),
        }
    }

    /// Write a value. Detects conflict if concurrent with existing value.
    pub fn write(
        &mut self,
        key: impl Into<String>,
        value: VersionedValue,
    ) -> Result<bool, ConflictError> {
        let k = key.into();
        let had_conflict;

        if let Some(existing) = self.store.get(&k) {
            if existing.clock.is_concurrent(&value.clock) {
                // Conflict detected.
                self.stats.conflicts_detected += 1;
                had_conflict = true;

                let resolved_value = match self.strategy {
                    ConflictStrategy::LastWriterWins => {
                        self.stats.auto_resolved += 1;
                        if value.timestamp_ms >= existing.timestamp_ms {
                            value.clone()
                        } else {
                            existing.clone()
                        }
                    }
                    ConflictStrategy::FirstWriterWins => {
                        self.stats.auto_resolved += 1;
                        if value.timestamp_ms <= existing.timestamp_ms {
                            value.clone()
                        } else {
                            existing.clone()
                        }
                    }
                    ConflictStrategy::Merge => {
                        self.stats.merge_attempts += 1;
                        let merged = self.try_merge(&existing.data, &value.data);
                        match merged {
                            Some(merged_data) => {
                                self.stats.merge_successes += 1;
                                self.stats.auto_resolved += 1;
                                let mut mv = value.clone();
                                mv.data = merged_data;
                                mv.clock.merge(&existing.clock);
                                mv
                            }
                            None => {
                                self.conflict_log.push(ConflictRecord {
                                    key: k.clone(),
                                    local_timestamp: existing.timestamp_ms,
                                    remote_timestamp: value.timestamp_ms,
                                    strategy: self.strategy,
                                    resolved: false,
                                    winner_node: None,
                                });
                                self.unresolved.insert(k.clone(), (existing.clone(), value.clone()));
                                return Err(ConflictError::MergeFailed {
                                    key: k,
                                    reason: "incompatible values".into(),
                                });
                            }
                        }
                    }
                    ConflictStrategy::Manual => {
                        self.unresolved.insert(k.clone(), (existing.clone(), value.clone()));
                        self.conflict_log.push(ConflictRecord {
                            key: k.clone(),
                            local_timestamp: existing.timestamp_ms,
                            remote_timestamp: value.timestamp_ms,
                            strategy: self.strategy,
                            resolved: false,
                            winner_node: None,
                        });
                        return Err(ConflictError::ManualResolutionRequired { key: k });
                    }
                };

                let winner = resolved_value.writer_node;
                self.conflict_log.push(ConflictRecord {
                    key: k.clone(),
                    local_timestamp: existing.timestamp_ms,
                    remote_timestamp: value.timestamp_ms,
                    strategy: self.strategy,
                    resolved: true,
                    winner_node: Some(winner),
                });

                let mut final_value = resolved_value;
                final_value.clock.merge(&existing.clock);
                self.store.insert(k, final_value);
            } else if value.clock.happened_before(&existing.clock) {
                // Stale update, ignore.
                had_conflict = false;
            } else {
                // Newer, apply directly.
                had_conflict = false;
                self.store.insert(k, value);
            }
        } else {
            had_conflict = false;
            self.store.insert(k, value);
        }

        Ok(had_conflict)
    }

    /// Simple merge: concatenate sorted unique bytes. Returns None if identical.
    fn try_merge(&self, a: &[u8], b: &[u8]) -> Option<Vec<u8>> {
        if a == b {
            return None;
        }
        let mut merged: Vec<u8> = a.iter().chain(b.iter()).copied().collect();
        merged.sort();
        merged.dedup();
        Some(merged)
    }

    /// Read a value.
    pub fn read(&self, key: &str) -> Option<&VersionedValue> {
        self.store.get(key)
    }

    /// Manually resolve a conflict by choosing a value.
    pub fn resolve_manual(
        &mut self,
        key: &str,
        chosen: VersionedValue,
    ) -> Result<(), ConflictError> {
        if !self.unresolved.contains_key(key) {
            return Err(ConflictError::KeyNotFound(key.to_string()));
        }
        self.unresolved.remove(key);
        self.stats.manual_resolved += 1;
        self.store.insert(key.to_string(), chosen);
        Ok(())
    }

    /// Three-way merge: given base, local, and remote, produce merged result.
    pub fn three_way_merge(
        &self,
        base: &[u8],
        local: &[u8],
        remote: &[u8],
    ) -> Result<Vec<u8>, ConflictError> {
        // Determine what each side changed relative to base.
        let local_added: Vec<u8> = local.iter().filter(|b| !base.contains(b)).copied().collect();
        let remote_added: Vec<u8> = remote.iter().filter(|b| !base.contains(b)).copied().collect();
        let local_removed: Vec<u8> = base.iter().filter(|b| !local.contains(b)).copied().collect();
        let remote_removed: Vec<u8> = base.iter().filter(|b| !remote.contains(b)).copied().collect();

        // Check for conflicting changes (both sides removed and added differently).
        for b in &local_added {
            if remote_removed.contains(b) {
                return Err(ConflictError::MergeFailed {
                    key: String::new(),
                    reason: format!("conflicting change on byte {b}"),
                });
            }
        }
        for b in &remote_added {
            if local_removed.contains(b) {
                return Err(ConflictError::MergeFailed {
                    key: String::new(),
                    reason: format!("conflicting change on byte {b}"),
                });
            }
        }

        // Merge: base + additions - removals.
        let mut result: Vec<u8> = base.iter()
            .filter(|b| !local_removed.contains(b) && !remote_removed.contains(b))
            .copied()
            .collect();
        result.extend(&local_added);
        result.extend(&remote_added);
        result.sort();
        result.dedup();
        Ok(result)
    }

    /// Unresolved conflict count.
    pub fn unresolved_count(&self) -> usize {
        self.unresolved.len()
    }

    /// Conflict log.
    pub fn conflict_log(&self) -> &[ConflictRecord] {
        &self.conflict_log
    }

    /// Number of entries in the store.
    pub fn store_size(&self) -> usize {
        self.store.len()
    }
}

impl fmt::Display for ConflictResolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConflictResolver(strategy={}, entries={}, unresolved={})",
            self.strategy,
            self.store.len(),
            self.unresolved.len(),
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_value(node: u64, ts: u64, data: Vec<u8>) -> VersionedValue {
        VersionedValue::new(data, node, ts)
    }

    #[test]
    fn vector_clock_increment() {
        let mut vc = VectorClock::new();
        vc.increment(1);
        assert_eq!(vc.get(1), 1);
        vc.increment(1);
        assert_eq!(vc.get(1), 2);
    }

    #[test]
    fn vector_clock_merge() {
        let mut a = VectorClock::new();
        a.increment(1);
        a.increment(1);
        let mut b = VectorClock::new();
        b.increment(2);
        a.merge(&b);
        assert_eq!(a.get(1), 2);
        assert_eq!(a.get(2), 1);
    }

    #[test]
    fn vector_clock_happened_before() {
        let mut a = VectorClock::new();
        a.increment(1);
        let mut b = VectorClock::new();
        b.increment(1);
        b.increment(1);
        assert!(a.happened_before(&b));
        assert!(!b.happened_before(&a));
    }

    #[test]
    fn vector_clock_concurrent() {
        let mut a = VectorClock::new();
        a.increment(1);
        let mut b = VectorClock::new();
        b.increment(2);
        assert!(a.is_concurrent(&b));
    }

    #[test]
    fn vector_clock_display() {
        let mut vc = VectorClock::new();
        vc.increment(1);
        let d = format!("{vc}");
        assert!(d.contains("VClock"));
    }

    #[test]
    fn conflict_strategy_display() {
        assert_eq!(format!("{}", ConflictStrategy::LastWriterWins), "last-writer-wins");
        assert_eq!(format!("{}", ConflictStrategy::Manual), "manual");
    }

    #[test]
    fn no_conflict_sequential_writes() {
        let mut cr = ConflictResolver::new(ConflictStrategy::LastWriterWins);
        let v1 = make_value(1, 100, vec![1]);
        cr.write("k", v1).unwrap();

        let mut v2 = make_value(1, 200, vec![2]);
        v2.clock.increment(1); // v2 happened-after v1
        let conflict = cr.write("k", v2).unwrap();
        assert!(!conflict);
    }

    #[test]
    fn last_writer_wins_conflict() {
        let mut cr = ConflictResolver::new(ConflictStrategy::LastWriterWins);
        let v1 = make_value(1, 100, vec![1]);
        cr.write("k", v1).unwrap();
        let v2 = make_value(2, 200, vec![2]); // concurrent (different node)
        let conflict = cr.write("k", v2).unwrap();
        assert!(conflict);
        assert_eq!(cr.read("k").unwrap().data, vec![2]); // higher ts wins
    }

    #[test]
    fn first_writer_wins_conflict() {
        let mut cr = ConflictResolver::new(ConflictStrategy::FirstWriterWins);
        let v1 = make_value(1, 100, vec![1]);
        cr.write("k", v1).unwrap();
        let v2 = make_value(2, 200, vec![2]);
        cr.write("k", v2).unwrap();
        assert_eq!(cr.read("k").unwrap().data, vec![1]); // lower ts wins
    }

    #[test]
    fn merge_strategy() {
        let mut cr = ConflictResolver::new(ConflictStrategy::Merge);
        let v1 = make_value(1, 100, vec![1, 3]);
        cr.write("k", v1).unwrap();
        let v2 = make_value(2, 200, vec![2, 4]);
        cr.write("k", v2).unwrap();
        let data = &cr.read("k").unwrap().data;
        assert!(data.contains(&1));
        assert!(data.contains(&2));
    }

    #[test]
    fn manual_strategy_returns_error() {
        let mut cr = ConflictResolver::new(ConflictStrategy::Manual);
        let v1 = make_value(1, 100, vec![1]);
        cr.write("k", v1).unwrap();
        let v2 = make_value(2, 200, vec![2]);
        assert!(matches!(cr.write("k", v2), Err(ConflictError::ManualResolutionRequired { .. })));
    }

    #[test]
    fn manual_resolve() {
        let mut cr = ConflictResolver::new(ConflictStrategy::Manual);
        let v1 = make_value(1, 100, vec![1]);
        cr.write("k", v1).unwrap();
        let v2 = make_value(2, 200, vec![2]);
        let _ = cr.write("k", v2);
        cr.resolve_manual("k", make_value(1, 300, vec![99])).unwrap();
        assert_eq!(cr.read("k").unwrap().data, vec![99]);
        assert_eq!(cr.unresolved_count(), 0);
    }

    #[test]
    fn three_way_merge_additions() {
        let cr = ConflictResolver::new(ConflictStrategy::Merge);
        let base = vec![1, 2, 3];
        let local = vec![1, 2, 3, 4];
        let remote = vec![1, 2, 3, 5];
        let merged = cr.three_way_merge(&base, &local, &remote).unwrap();
        assert!(merged.contains(&4));
        assert!(merged.contains(&5));
    }

    #[test]
    fn three_way_merge_removals() {
        let cr = ConflictResolver::new(ConflictStrategy::Merge);
        let base = vec![1, 2, 3];
        let local = vec![1, 3]; // removed 2
        let remote = vec![1, 2, 3];
        let merged = cr.three_way_merge(&base, &local, &remote).unwrap();
        assert!(!merged.contains(&2));
    }

    #[test]
    fn conflict_log_recorded() {
        let mut cr = ConflictResolver::new(ConflictStrategy::LastWriterWins);
        cr.write("k", make_value(1, 100, vec![1])).unwrap();
        cr.write("k", make_value(2, 200, vec![2])).unwrap();
        assert!(!cr.conflict_log().is_empty());
    }

    #[test]
    fn resolution_stats_tracked() {
        let mut cr = ConflictResolver::new(ConflictStrategy::LastWriterWins);
        cr.write("k", make_value(1, 100, vec![1])).unwrap();
        cr.write("k", make_value(2, 200, vec![2])).unwrap();
        assert_eq!(cr.stats.conflicts_detected, 1);
        assert_eq!(cr.stats.auto_resolved, 1);
    }

    #[test]
    fn store_size() {
        let mut cr = ConflictResolver::new(ConflictStrategy::LastWriterWins);
        cr.write("a", make_value(1, 100, vec![1])).unwrap();
        cr.write("b", make_value(1, 100, vec![2])).unwrap();
        assert_eq!(cr.store_size(), 2);
    }

    #[test]
    fn resolver_display() {
        let cr = ConflictResolver::new(ConflictStrategy::LastWriterWins);
        let d = format!("{cr}");
        assert!(d.contains("ConflictResolver"));
    }

    #[test]
    fn conflict_record_display() {
        let rec = ConflictRecord {
            key: "test".into(),
            local_timestamp: 100,
            remote_timestamp: 200,
            strategy: ConflictStrategy::Merge,
            resolved: true,
            winner_node: Some(1),
        };
        let d = format!("{rec}");
        assert!(d.contains("Conflict"));
    }
}
