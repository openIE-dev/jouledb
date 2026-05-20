//! CRDT registers — LWW-Register (last-writer-wins), MV-Register (multi-value),
//! timestamp ordering, concurrent write resolution, register history, merge
//! from multiple replicas.

use std::collections::HashMap;

// ── LWW-Register (Last-Writer-Wins Register) ────────────────────────────────

/// A last-writer-wins register. The write with the highest timestamp wins.
/// Ties are broken by replica id (lexicographic ordering).
#[derive(Debug, Clone)]
pub struct LwwRegister {
    /// Current value.
    value: String,
    /// Timestamp of the current value.
    timestamp: u64,
    /// Replica that wrote the current value.
    writer: String,
    /// History of all writes (optional, for auditing).
    history: Vec<RegisterWrite>,
    /// Whether to keep history.
    keep_history: bool,
}

/// A single write event to a register.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterWrite {
    /// Value written.
    pub value: String,
    /// Timestamp of the write.
    pub timestamp: u64,
    /// Replica that performed the write.
    pub writer: String,
}

impl LwwRegister {
    /// Create a new LWW-Register with an initial value.
    pub fn new(value: &str, timestamp: u64, writer: &str) -> Self {
        let write = RegisterWrite {
            value: value.to_string(),
            timestamp,
            writer: writer.to_string(),
        };
        Self {
            value: value.to_string(),
            timestamp,
            writer: writer.to_string(),
            history: vec![write],
            keep_history: true,
        }
    }

    /// Create a new LWW-Register without history tracking.
    pub fn without_history(value: &str, timestamp: u64, writer: &str) -> Self {
        Self {
            value: value.to_string(),
            timestamp,
            writer: writer.to_string(),
            history: Vec::new(),
            keep_history: false,
        }
    }

    /// Get the current value.
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Get the current timestamp.
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Get the writer of the current value.
    pub fn writer(&self) -> &str {
        &self.writer
    }

    /// Write a new value. Only succeeds if the timestamp is higher than
    /// the current one (or equal timestamp with a higher replica id).
    /// Returns true if the write was accepted.
    pub fn write(&mut self, value: &str, timestamp: u64, writer: &str) -> bool {
        if self.should_accept(timestamp, writer) {
            self.value = value.to_string();
            self.timestamp = timestamp;
            self.writer = writer.to_string();
            if self.keep_history {
                self.history.push(RegisterWrite {
                    value: value.to_string(),
                    timestamp,
                    writer: writer.to_string(),
                });
            }
            true
        } else {
            false
        }
    }

    /// Check if a new write should be accepted.
    fn should_accept(&self, timestamp: u64, writer: &str) -> bool {
        if timestamp > self.timestamp {
            return true;
        }
        if timestamp == self.timestamp {
            return writer > self.writer.as_str();
        }
        false
    }

    /// Merge with another LWW-Register. The register with the higher
    /// timestamp wins.
    pub fn merge(&mut self, other: &LwwRegister) {
        if other.timestamp > self.timestamp
            || (other.timestamp == self.timestamp && other.writer > self.writer)
        {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
            self.writer = other.writer.clone();
        }
        // Merge history if both keep it.
        if self.keep_history && other.keep_history {
            for w in &other.history {
                let dominated = self.history.iter().any(|h| h == w);
                if !dominated {
                    self.history.push(w.clone());
                }
            }
            self.history.sort_by(|a, b| {
                a.timestamp.cmp(&b.timestamp)
                    .then(a.writer.cmp(&b.writer))
            });
        }
    }

    /// Get the write history (newest last).
    pub fn history(&self) -> &[RegisterWrite] {
        &self.history
    }

    /// Get the number of writes in history.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

// ── MV-Register (Multi-Value Register) ──────────────────────────────────────

/// A single version in the MV-Register, tagged with a vector clock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvValue {
    /// The stored value.
    pub value: String,
    /// The vector clock representing causal context.
    pub clock: HashMap<String, u64>,
}

/// A multi-value register that preserves all concurrent writes.
/// Uses vector clocks to detect concurrency. On merge, concurrent values
/// are all kept; dominated values are pruned.
#[derive(Debug, Clone)]
pub struct MvRegister {
    /// Replica id.
    replica_id: String,
    /// Current set of concurrent values.
    values: Vec<MvValue>,
    /// This replica's logical clock (for generating vector timestamps).
    local_clock: HashMap<String, u64>,
}

impl MvRegister {
    /// Create a new empty MV-Register.
    pub fn new(replica_id: &str) -> Self {
        Self {
            replica_id: replica_id.to_string(),
            values: Vec::new(),
            local_clock: HashMap::new(),
        }
    }

    /// Write a new value. This supersedes all currently observed values.
    pub fn write(&mut self, value: &str) {
        // Advance the local clock.
        let entry = self.local_clock.entry(self.replica_id.clone()).or_insert(0);
        *entry += 1;
        // Merge all existing clocks into the local clock.
        for v in &self.values {
            for (rid, &ts) in &v.clock {
                let e = self.local_clock.entry(rid.clone()).or_insert(0);
                if ts > *e {
                    *e = ts;
                }
            }
        }
        // Replace all values with this single new value.
        self.values = vec![MvValue {
            value: value.to_string(),
            clock: self.local_clock.clone(),
        }];
    }

    /// Get the current values. If there is no concurrency, this returns a
    /// single value. If concurrent writes exist, all are returned.
    pub fn values(&self) -> Vec<&str> {
        self.values.iter().map(|v| v.value.as_str()).collect()
    }

    /// Get the number of concurrent values.
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// Check if there are concurrent values (conflict).
    pub fn has_conflict(&self) -> bool {
        self.values.len() > 1
    }

    /// Get the replica id.
    pub fn replica_id(&self) -> &str {
        &self.replica_id
    }

    /// Merge with another MV-Register.
    /// Keeps all values that are not dominated by another.
    pub fn merge(&mut self, other: &MvRegister) {
        let mut all_values: Vec<MvValue> = Vec::new();
        // Collect all values from both registers.
        for v in &self.values {
            all_values.push(v.clone());
        }
        for v in &other.values {
            all_values.push(v.clone());
        }
        // Remove dominated values.
        let mut result: Vec<MvValue> = Vec::new();
        for (i, v) in all_values.iter().enumerate() {
            let is_dominated = all_values.iter().enumerate().any(|(j, u)| {
                i != j && Self::dominates(&u.clock, &v.clock)
            });
            if !is_dominated {
                // Avoid duplicates.
                let already_present = result.iter().any(|r| r == v);
                if !already_present {
                    result.push(v.clone());
                }
            }
        }
        self.values = result;
        // Merge local clocks.
        for (rid, &ts) in &other.local_clock {
            let entry = self.local_clock.entry(rid.clone()).or_insert(0);
            if ts > *entry {
                *entry = ts;
            }
        }
    }

    /// Check if clock `a` strictly dominates clock `b`
    /// (all entries in a >= corresponding entries in b, and at least one >).
    fn dominates(a: &HashMap<String, u64>, b: &HashMap<String, u64>) -> bool {
        let mut dominated = true;
        let mut strictly_greater = false;
        // Every entry in b must be <= corresponding entry in a.
        for (rid, &bval) in b {
            let aval = a.get(rid).copied().unwrap_or(0);
            if aval < bval {
                dominated = false;
                break;
            }
            if aval > bval {
                strictly_greater = true;
            }
        }
        if !dominated {
            return false;
        }
        // Check entries in a that are not in b.
        if !strictly_greater {
            for (rid, &aval) in a {
                if !b.contains_key(rid) && aval > 0 {
                    strictly_greater = true;
                    break;
                }
            }
        }
        strictly_greater
    }

    /// Resolve a conflict by picking one of the concurrent values.
    /// This is a manual conflict resolution step.
    pub fn resolve(&mut self, chosen_value: &str) {
        self.write(chosen_value);
    }

    /// Get the internal vector clocks for all concurrent values.
    pub fn clocks(&self) -> Vec<&HashMap<String, u64>> {
        self.values.iter().map(|v| &v.clock).collect()
    }

    /// Check if the register is empty (no writes).
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LWW-Register tests ──

    #[test]
    fn lww_new_and_read() {
        let r = LwwRegister::new("hello", 1, "r1");
        assert_eq!(r.value(), "hello");
        assert_eq!(r.timestamp(), 1);
        assert_eq!(r.writer(), "r1");
    }

    #[test]
    fn lww_write_higher_timestamp() {
        let mut r = LwwRegister::new("a", 1, "r1");
        assert!(r.write("b", 2, "r1"));
        assert_eq!(r.value(), "b");
    }

    #[test]
    fn lww_write_lower_timestamp_rejected() {
        let mut r = LwwRegister::new("a", 5, "r1");
        assert!(!r.write("b", 3, "r2"));
        assert_eq!(r.value(), "a");
    }

    #[test]
    fn lww_write_same_timestamp_higher_replica_wins() {
        let mut r = LwwRegister::new("a", 5, "r1");
        assert!(r.write("b", 5, "r2")); // "r2" > "r1".
        assert_eq!(r.value(), "b");
    }

    #[test]
    fn lww_write_same_timestamp_lower_replica_rejected() {
        let mut r = LwwRegister::new("a", 5, "r2");
        assert!(!r.write("b", 5, "r1")); // "r1" < "r2".
        assert_eq!(r.value(), "a");
    }

    #[test]
    fn lww_merge_other_wins() {
        let mut r1 = LwwRegister::new("old", 1, "r1");
        let r2 = LwwRegister::new("new", 2, "r2");
        r1.merge(&r2);
        assert_eq!(r1.value(), "new");
        assert_eq!(r1.timestamp(), 2);
    }

    #[test]
    fn lww_merge_self_wins() {
        let mut r1 = LwwRegister::new("mine", 5, "r1");
        let r2 = LwwRegister::new("theirs", 3, "r2");
        r1.merge(&r2);
        assert_eq!(r1.value(), "mine");
    }

    #[test]
    fn lww_merge_tie_broken_by_replica() {
        let mut r1 = LwwRegister::new("val1", 5, "r1");
        let r2 = LwwRegister::new("val2", 5, "r2");
        r1.merge(&r2);
        assert_eq!(r1.value(), "val2"); // "r2" > "r1".
    }

    #[test]
    fn lww_history_tracked() {
        let mut r = LwwRegister::new("a", 1, "r1");
        r.write("b", 2, "r1");
        r.write("c", 3, "r2");
        assert_eq!(r.history_len(), 3);
        let h = r.history();
        assert_eq!(h[0].value, "a");
        assert_eq!(h[1].value, "b");
        assert_eq!(h[2].value, "c");
    }

    #[test]
    fn lww_without_history() {
        let mut r = LwwRegister::without_history("a", 1, "r1");
        r.write("b", 2, "r1");
        assert_eq!(r.history_len(), 0);
        assert_eq!(r.value(), "b");
    }

    // ── MV-Register tests ──

    #[test]
    fn mv_new_is_empty() {
        let r = MvRegister::new("r1");
        assert!(r.is_empty());
        assert_eq!(r.value_count(), 0);
    }

    #[test]
    fn mv_single_write() {
        let mut r = MvRegister::new("r1");
        r.write("hello");
        assert_eq!(r.values(), vec!["hello"]);
        assert!(!r.has_conflict());
    }

    #[test]
    fn mv_sequential_writes_overwrite() {
        let mut r = MvRegister::new("r1");
        r.write("a");
        r.write("b");
        assert_eq!(r.values(), vec!["b"]);
    }

    #[test]
    fn mv_concurrent_writes_create_conflict() {
        let mut r1 = MvRegister::new("r1");
        let mut r2 = MvRegister::new("r2");
        r1.write("value_from_r1");
        r2.write("value_from_r2");
        r1.merge(&r2);
        assert!(r1.has_conflict());
        assert_eq!(r1.value_count(), 2);
        let mut vals = r1.values();
        vals.sort();
        assert_eq!(vals, vec!["value_from_r1", "value_from_r2"]);
    }

    #[test]
    fn mv_dominated_value_pruned() {
        let mut r1 = MvRegister::new("r1");
        let mut r2 = MvRegister::new("r2");
        r1.write("a");
        // r2 knows about r1's write and then writes.
        r2.merge(&r1);
        r2.write("b"); // This dominates "a".
        r1.merge(&r2);
        assert!(!r1.has_conflict());
        assert_eq!(r1.values(), vec!["b"]);
    }

    #[test]
    fn mv_resolve_conflict() {
        let mut r1 = MvRegister::new("r1");
        let mut r2 = MvRegister::new("r2");
        r1.write("x");
        r2.write("y");
        r1.merge(&r2);
        assert!(r1.has_conflict());
        r1.resolve("z");
        assert!(!r1.has_conflict());
        assert_eq!(r1.values(), vec!["z"]);
    }

    #[test]
    fn mv_three_way_concurrent() {
        let mut r1 = MvRegister::new("r1");
        let mut r2 = MvRegister::new("r2");
        let mut r3 = MvRegister::new("r3");
        r1.write("a");
        r2.write("b");
        r3.write("c");
        r1.merge(&r2);
        r1.merge(&r3);
        assert_eq!(r1.value_count(), 3);
    }

    #[test]
    fn mv_clocks() {
        let mut r = MvRegister::new("r1");
        r.write("hello");
        let clocks = r.clocks();
        assert_eq!(clocks.len(), 1);
        assert_eq!(clocks[0].get("r1"), Some(&1));
    }

    #[test]
    fn mv_merge_preserves_local_clock() {
        let mut r1 = MvRegister::new("r1");
        let mut r2 = MvRegister::new("r2");
        r1.write("a");
        r2.write("b");
        r1.merge(&r2);
        // After merge, r1's local clock should include r2.
        r1.write("c"); // This should dominate both "a" and "b".
        assert!(!r1.has_conflict());
    }

    #[test]
    fn dominates_function() {
        let mut a: HashMap<String, u64> = HashMap::new();
        a.insert("r1".to_string(), 2);
        a.insert("r2".to_string(), 1);
        let mut b: HashMap<String, u64> = HashMap::new();
        b.insert("r1".to_string(), 1);
        assert!(MvRegister::dominates(&a, &b));
        assert!(!MvRegister::dominates(&b, &a));
    }

    #[test]
    fn dominates_equal_clocks_not_dominated() {
        let mut a: HashMap<String, u64> = HashMap::new();
        a.insert("r1".to_string(), 1);
        let b = a.clone();
        assert!(!MvRegister::dominates(&a, &b));
    }
}
