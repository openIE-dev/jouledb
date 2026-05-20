//! Temporal/bitemporal indexing — valid-time and transaction-time tracking.
//!
//! Replaces temporal database extensions with a pure-Rust bitemporal index.
//! Supports as-of queries, time-range queries, temporal joins, snapshots,
//! per-key history, and temporal aggregation over both valid time and
//! transaction time dimensions.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

// ── Time Range ────────────────────────────────────────────────

/// A time range [start, end) where end is exclusive. Uses millisecond timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimeRange {
    /// Inclusive start (ms).
    pub start: u64,
    /// Exclusive end (ms). `u64::MAX` means open-ended (still valid).
    pub end: u64,
}

impl TimeRange {
    /// Create a closed range [start, end).
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    /// Create an open-ended range [start, MAX).
    pub fn from(start: u64) -> Self {
        Self {
            start,
            end: u64::MAX,
        }
    }

    /// Does this range contain the given timestamp?
    pub fn contains(&self, ts: u64) -> bool {
        ts >= self.start && ts < self.end
    }

    /// Do two ranges overlap?
    pub fn overlaps(&self, other: &TimeRange) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Duration of this range in ms (saturating for open-ended).
    pub fn duration_ms(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Is this range open-ended (still valid)?
    pub fn is_open(&self) -> bool {
        self.end == u64::MAX
    }

    /// Close this range at the given timestamp.
    pub fn close_at(&mut self, ts: u64) {
        if ts < self.end {
            self.end = ts;
        }
    }
}

// ── Bitemporal Record ─────────────────────────────────────────

/// A record with both valid-time and transaction-time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitemporalRecord {
    pub key: String,
    pub value: String,
    /// When the fact was true in the real world.
    pub valid_time: TimeRange,
    /// When the fact was recorded in the system.
    pub transaction_time: TimeRange,
    /// Version number for this key.
    pub version: u64,
}

// ── Temporal Aggregate ────────────────────────────────────────

/// Result of a temporal aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAggregate {
    pub time_range: TimeRange,
    pub count: u64,
    pub keys: Vec<String>,
}

// ── Temporal Index ────────────────────────────────────────────

/// Bitemporal index supporting valid-time and transaction-time queries.
#[derive(Debug)]
pub struct TemporalIndex {
    /// All records, ordered by transaction time start.
    records: Vec<BitemporalRecord>,
    /// Key -> list of record indices, ordered by valid_time.start.
    key_index: HashMap<String, Vec<usize>>,
    /// Transaction time -> record indices (BTree for range queries).
    tx_index: BTreeMap<u64, Vec<usize>>,
    /// Per-key version counter.
    versions: HashMap<String, u64>,
    /// Current transaction timestamp.
    current_tx_time: u64,
}

impl TemporalIndex {
    /// Create a new temporal index.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            key_index: HashMap::new(),
            tx_index: BTreeMap::new(),
            versions: HashMap::new(),
            current_tx_time: 0,
        }
    }

    /// Set the current transaction time (used for inserts).
    pub fn set_transaction_time(&mut self, tx_time: u64) {
        self.current_tx_time = tx_time;
    }

    /// Insert a fact valid during [valid_start, valid_end).
    /// Records the transaction time as the current tx time.
    pub fn insert(
        &mut self,
        key: &str,
        value: &str,
        valid_start: u64,
        valid_end: u64,
    ) -> usize {
        let version = self.versions.entry(key.to_string()).or_insert(0);
        *version += 1;

        let record = BitemporalRecord {
            key: key.to_string(),
            value: value.to_string(),
            valid_time: TimeRange::new(valid_start, valid_end),
            transaction_time: TimeRange::from(self.current_tx_time),
            version: *version,
        };

        let idx = self.records.len();
        self.records.push(record);
        self.key_index
            .entry(key.to_string())
            .or_default()
            .push(idx);
        self.tx_index
            .entry(self.current_tx_time)
            .or_default()
            .push(idx);
        idx
    }

    /// Insert a fact valid from now with no end (open-ended).
    pub fn insert_open(&mut self, key: &str, value: &str, valid_start: u64) -> usize {
        self.insert(key, value, valid_start, u64::MAX)
    }

    /// Update a key: close the current version's valid time and insert a new one.
    pub fn update(
        &mut self,
        key: &str,
        new_value: &str,
        valid_start: u64,
        valid_end: u64,
    ) -> usize {
        // Close all open records for this key at valid_start.
        if let Some(indices) = self.key_index.get(key) {
            let indices_clone: Vec<usize> = indices.clone();
            for idx in indices_clone {
                if self.records[idx].valid_time.is_open()
                    && self.records[idx].transaction_time.is_open()
                {
                    self.records[idx].valid_time.close_at(valid_start);
                }
            }
        }
        self.insert(key, new_value, valid_start, valid_end)
    }

    /// Delete a key by closing all open records at the given time.
    pub fn delete(&mut self, key: &str, at_time: u64) {
        if let Some(indices) = self.key_index.get(key) {
            let indices_clone: Vec<usize> = indices.clone();
            for idx in indices_clone {
                let rec = &mut self.records[idx];
                if rec.valid_time.is_open() && rec.transaction_time.is_open() {
                    rec.valid_time.close_at(at_time);
                    rec.transaction_time.close_at(self.current_tx_time);
                }
            }
        }
    }

    /// As-of query: what was the value of `key` at valid_time `vt`?
    /// Returns the most recent version visible at that time.
    pub fn as_of(&self, key: &str, vt: u64) -> Option<&BitemporalRecord> {
        self.key_index.get(key).and_then(|indices| {
            indices
                .iter()
                .rev()
                .filter_map(|idx| {
                    let rec = &self.records[*idx];
                    if rec.valid_time.contains(vt) {
                        Some(rec)
                    } else {
                        None
                    }
                })
                .next()
        })
    }

    /// Bitemporal as-of: value of `key` at valid_time `vt` as known at transaction_time `tt`.
    pub fn as_of_bitemporal(
        &self,
        key: &str,
        vt: u64,
        tt: u64,
    ) -> Option<&BitemporalRecord> {
        self.key_index.get(key).and_then(|indices| {
            indices
                .iter()
                .rev()
                .filter_map(|idx| {
                    let rec = &self.records[*idx];
                    if rec.valid_time.contains(vt)
                        && rec.transaction_time.contains(tt)
                    {
                        Some(rec)
                    } else {
                        None
                    }
                })
                .next()
        })
    }

    /// Time-range query: all records for `key` that overlap the given valid time range.
    pub fn range_query(
        &self,
        key: &str,
        range: &TimeRange,
    ) -> Vec<&BitemporalRecord> {
        self.key_index
            .get(key)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|idx| {
                        let rec = &self.records[*idx];
                        if rec.valid_time.overlaps(range) {
                            Some(rec)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Snapshot at time T: all current values at valid_time `vt`.
    pub fn snapshot(&self, vt: u64) -> Vec<&BitemporalRecord> {
        let mut result: HashMap<&str, &BitemporalRecord> = HashMap::new();

        for rec in &self.records {
            if rec.valid_time.contains(vt) {
                let current = result.get(rec.key.as_str());
                let should_replace = match current {
                    None => true,
                    Some(existing) => rec.version > existing.version,
                };
                if should_replace {
                    result.insert(&rec.key, rec);
                }
            }
        }

        let mut records: Vec<&BitemporalRecord> = result.into_values().collect();
        records.sort_by(|a, b| a.key.cmp(&b.key));
        records
    }

    /// Full history for a key, ordered by valid_time start.
    pub fn history(&self, key: &str) -> Vec<&BitemporalRecord> {
        self.key_index
            .get(key)
            .map(|indices| {
                let mut recs: Vec<&BitemporalRecord> = indices
                    .iter()
                    .map(|idx| &self.records[*idx])
                    .collect();
                recs.sort_by_key(|r| r.valid_time.start);
                recs
            })
            .unwrap_or_default()
    }

    /// Temporal join: find all pairs (from left_key, right_key) where
    /// their valid times overlap.
    pub fn temporal_join(
        &self,
        left_key: &str,
        right_key: &str,
    ) -> Vec<(&BitemporalRecord, &BitemporalRecord)> {
        let left = self.history(left_key);
        let right = self.history(right_key);
        let mut pairs = Vec::new();

        for l in &left {
            for r in &right {
                if l.valid_time.overlaps(&r.valid_time) {
                    pairs.push((*l, *r));
                }
            }
        }

        pairs
    }

    /// Temporal aggregation: count records per time bucket.
    pub fn aggregate(&self, bucket_size_ms: u64, range: &TimeRange) -> Vec<TemporalAggregate> {
        let mut buckets: BTreeMap<u64, (u64, Vec<String>)> = BTreeMap::new();

        let mut ts = range.start;
        while ts < range.end {
            buckets.insert(ts, (0, Vec::new()));
            ts = ts.saturating_add(bucket_size_ms);
        }

        for rec in &self.records {
            for (&bucket_start, value) in buckets.iter_mut() {
                let bucket_range = TimeRange::new(bucket_start, bucket_start + bucket_size_ms);
                if rec.valid_time.overlaps(&bucket_range) {
                    value.0 += 1;
                    if !value.1.contains(&rec.key) {
                        value.1.push(rec.key.clone());
                    }
                }
            }
        }

        buckets
            .into_iter()
            .map(|(start, (count, keys))| TemporalAggregate {
                time_range: TimeRange::new(start, start + bucket_size_ms),
                count,
                keys,
            })
            .collect()
    }

    /// Total number of records in the index.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Number of distinct keys.
    pub fn key_count(&self) -> usize {
        self.key_index.len()
    }

    /// Get all distinct keys.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.key_index.keys().cloned().collect();
        keys.sort();
        keys
    }
}

impl Default for TemporalIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_range_contains() {
        let r = TimeRange::new(100, 200);
        assert!(r.contains(100));
        assert!(r.contains(150));
        assert!(!r.contains(200)); // Exclusive end.
        assert!(!r.contains(50));
    }

    #[test]
    fn test_time_range_overlaps() {
        let a = TimeRange::new(100, 200);
        let b = TimeRange::new(150, 250);
        assert!(a.overlaps(&b));
        let c = TimeRange::new(200, 300);
        assert!(!a.overlaps(&c)); // Adjacent, no overlap.
    }

    #[test]
    fn test_time_range_open_ended() {
        let r = TimeRange::from(100);
        assert!(r.is_open());
        assert!(r.contains(u64::MAX - 1));
    }

    #[test]
    fn test_time_range_close() {
        let mut r = TimeRange::from(100);
        r.close_at(500);
        assert!(!r.is_open());
        assert_eq!(r.end, 500);
    }

    #[test]
    fn test_insert_and_as_of() {
        let mut idx = TemporalIndex::new();
        idx.insert("temp", "72F", 1000, 2000);
        let rec = idx.as_of("temp", 1500).unwrap();
        assert_eq!(rec.value, "72F");
    }

    #[test]
    fn test_as_of_outside_range() {
        let mut idx = TemporalIndex::new();
        idx.insert("temp", "72F", 1000, 2000);
        assert!(idx.as_of("temp", 2500).is_none());
        assert!(idx.as_of("temp", 500).is_none());
    }

    #[test]
    fn test_multiple_versions() {
        let mut idx = TemporalIndex::new();
        idx.insert("temp", "72F", 1000, 2000);
        idx.insert("temp", "75F", 2000, 3000);
        assert_eq!(idx.as_of("temp", 1500).unwrap().value, "72F");
        assert_eq!(idx.as_of("temp", 2500).unwrap().value, "75F");
    }

    #[test]
    fn test_history() {
        let mut idx = TemporalIndex::new();
        idx.insert("temp", "72F", 1000, 2000);
        idx.insert("temp", "75F", 2000, 3000);
        idx.insert("temp", "70F", 3000, 4000);
        let hist = idx.history("temp");
        assert_eq!(hist.len(), 3);
        assert_eq!(hist[0].value, "72F");
        assert_eq!(hist[2].value, "70F");
    }

    #[test]
    fn test_snapshot() {
        let mut idx = TemporalIndex::new();
        idx.insert("temp", "72F", 1000, 3000);
        idx.insert("humidity", "50%", 1000, 3000);
        idx.insert("pressure", "1013hPa", 2000, 4000);
        let snap = idx.snapshot(2500);
        assert_eq!(snap.len(), 3);
    }

    #[test]
    fn test_range_query() {
        let mut idx = TemporalIndex::new();
        idx.insert("temp", "72F", 1000, 2000);
        idx.insert("temp", "75F", 2000, 3000);
        idx.insert("temp", "70F", 5000, 6000);
        let range = TimeRange::new(1500, 2500);
        let results = idx.range_query("temp", &range);
        assert_eq!(results.len(), 2); // First two overlap.
    }

    #[test]
    fn test_bitemporal_as_of() {
        let mut idx = TemporalIndex::new();
        idx.set_transaction_time(100);
        idx.insert("temp", "72F", 1000, 2000);
        idx.set_transaction_time(200);
        idx.insert("temp", "75F", 1000, 2000); // Correction at tx=200.
        // As-of vt=1500, as known at tx=150 -> should see 72F.
        let rec = idx.as_of_bitemporal("temp", 1500, 150).unwrap();
        assert_eq!(rec.value, "72F");
        // As-of vt=1500, as known at tx=250 -> should see 75F (most recent version).
        let rec = idx.as_of_bitemporal("temp", 1500, 250).unwrap();
        assert_eq!(rec.value, "75F");
    }

    #[test]
    fn test_temporal_join() {
        let mut idx = TemporalIndex::new();
        idx.insert("employee", "Alice", 1000, 3000);
        idx.insert("project", "Alpha", 2000, 4000);
        let pairs = idx.temporal_join("employee", "project");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0.value, "Alice");
        assert_eq!(pairs[0].1.value, "Alpha");
    }

    #[test]
    fn test_temporal_join_no_overlap() {
        let mut idx = TemporalIndex::new();
        idx.insert("a", "v1", 1000, 2000);
        idx.insert("b", "v2", 3000, 4000);
        let pairs = idx.temporal_join("a", "b");
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_update_closes_previous() {
        let mut idx = TemporalIndex::new();
        idx.insert_open("temp", "72F", 1000);
        idx.update("temp", "75F", 2000, u64::MAX);
        let hist = idx.history("temp");
        assert_eq!(hist.len(), 2);
        // First should be closed at 2000.
        assert_eq!(hist[0].valid_time.end, 2000);
        assert!(hist[1].valid_time.is_open());
    }

    #[test]
    fn test_delete() {
        let mut idx = TemporalIndex::new();
        idx.insert_open("temp", "72F", 1000);
        idx.delete("temp", 2000);
        assert!(idx.as_of("temp", 2500).is_none());
        assert_eq!(idx.as_of("temp", 1500).unwrap().value, "72F");
    }

    #[test]
    fn test_aggregate() {
        let mut idx = TemporalIndex::new();
        idx.insert("a", "1", 0, 500);
        idx.insert("b", "2", 200, 800);
        idx.insert("c", "3", 600, 1000);
        let range = TimeRange::new(0, 1000);
        let agg = idx.aggregate(500, &range);
        assert_eq!(agg.len(), 2);
        // Bucket [0,500): a and b overlap.
        assert_eq!(agg[0].count, 2);
        // Bucket [500,1000): b and c overlap.
        assert_eq!(agg[1].count, 2);
    }

    #[test]
    fn test_key_count() {
        let mut idx = TemporalIndex::new();
        idx.insert("a", "1", 0, 100);
        idx.insert("b", "2", 0, 100);
        idx.insert("a", "3", 100, 200);
        assert_eq!(idx.key_count(), 2);
        assert_eq!(idx.record_count(), 3);
    }

    #[test]
    fn test_keys_sorted() {
        let mut idx = TemporalIndex::new();
        idx.insert("charlie", "c", 0, 100);
        idx.insert("alpha", "a", 0, 100);
        idx.insert("bravo", "b", 0, 100);
        let keys = idx.keys();
        assert_eq!(keys, vec!["alpha", "bravo", "charlie"]);
    }

    #[test]
    fn test_default_constructor() {
        let idx = TemporalIndex::default();
        assert_eq!(idx.record_count(), 0);
        assert_eq!(idx.key_count(), 0);
    }

    #[test]
    fn test_time_range_duration() {
        let r = TimeRange::new(100, 350);
        assert_eq!(r.duration_ms(), 250);
    }
}
