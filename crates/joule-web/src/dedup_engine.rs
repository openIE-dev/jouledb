//! Deduplication engine.
//!
//! Replaces `pandas.drop_duplicates`, `lodash.uniqBy`, and similar dedup libraries
//! with a pure-Rust engine. Supports exact match dedup, fuzzy dedup (edit distance
//! threshold), composite key dedup, first/last/merge strategies for duplicate
//! resolution, dedup statistics, and sliding window dedup for streams.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors from the dedup engine.
#[derive(Debug, Clone, PartialEq)]
pub enum DedupError {
    /// No key fields configured.
    NoKeyFields,
    /// Field not found in record.
    FieldNotFound { record_index: usize, field: String },
    /// Invalid configuration.
    InvalidConfig(String),
}

impl fmt::Display for DedupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoKeyFields => write!(f, "no key fields configured"),
            Self::FieldNotFound { record_index, field } => {
                write!(f, "field '{field}' not found in record {record_index}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for DedupError {}

// ── Record type ──────────────────────────────────────────────────

/// A data record for deduplication.
pub type Row = HashMap<String, serde_json::Value>;

// ── Resolution strategy ──────────────────────────────────────────

/// How to resolve when duplicates are found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    /// Keep the first occurrence.
    KeepFirst,
    /// Keep the last occurrence.
    KeepLast,
    /// Merge fields from all duplicates (last value wins for conflicts).
    MergeLast,
    /// Merge fields from all duplicates (first value wins for conflicts).
    MergeFirst,
}

impl Default for ResolutionStrategy {
    fn default() -> Self {
        Self::KeepFirst
    }
}

// ── Match mode ───────────────────────────────────────────────────

/// How to determine if two records are duplicates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchMode {
    /// Exact match on key fields.
    Exact,
    /// Fuzzy match using edit distance on string fields.
    Fuzzy { max_distance: usize },
}

impl Default for MatchMode {
    fn default() -> Self {
        Self::Exact
    }
}

// ── Dedup statistics ─────────────────────────────────────────────

/// Statistics from a dedup run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DedupStats {
    /// Total records processed.
    pub total_input: usize,
    /// Unique records in output.
    pub total_output: usize,
    /// Number of duplicates found.
    pub duplicates_found: usize,
    /// Number of duplicate groups (sets of records sharing a key).
    pub duplicate_groups: usize,
    /// Size of the largest duplicate group.
    pub largest_group: usize,
    /// Dedup ratio (duplicates / total).
    pub dedup_ratio: f64,
}

impl DedupStats {
    fn compute(total_input: usize, total_output: usize, group_sizes: &[usize]) -> Self {
        let duplicates_found = total_input.saturating_sub(total_output);
        let duplicate_groups = group_sizes.iter().filter(|&&s| s > 1).count();
        let largest_group = group_sizes.iter().copied().max().unwrap_or(0);
        let dedup_ratio = if total_input == 0 {
            0.0
        } else {
            duplicates_found as f64 / total_input as f64
        };

        Self {
            total_input,
            total_output,
            duplicates_found,
            duplicate_groups,
            largest_group,
            dedup_ratio,
        }
    }
}

// ── Dedup result ─────────────────────────────────────────────────

/// Result of a dedup operation.
#[derive(Debug, Clone)]
pub struct DedupResult {
    /// Deduplicated records.
    pub records: Vec<Row>,
    /// Records that were identified as duplicates.
    pub duplicates: Vec<Row>,
    /// Statistics.
    pub stats: DedupStats,
}

// ── Sliding window entry ─────────────────────────────────────────

/// An entry in the sliding window for stream dedup.
struct WindowEntry {
    key: String,
    record: Row,
    sequence: u64,
}

// ── DedupEngine ──────────────────────────────────────────────────

/// The deduplication engine.
#[derive(Debug, Clone)]
pub struct DedupEngine {
    /// Key fields used for matching.
    key_fields: Vec<String>,
    /// Match mode.
    match_mode: MatchMode,
    /// Resolution strategy.
    strategy: ResolutionStrategy,
    /// Case-insensitive matching for string fields.
    case_insensitive: bool,
}

impl DedupEngine {
    /// Create a new dedup engine with the given key fields.
    pub fn new(key_fields: Vec<String>) -> Self {
        Self {
            key_fields,
            match_mode: MatchMode::default(),
            strategy: ResolutionStrategy::default(),
            case_insensitive: false,
        }
    }

    /// Set the match mode.
    pub fn with_match_mode(mut self, mode: MatchMode) -> Self {
        self.match_mode = mode;
        self
    }

    /// Set the resolution strategy.
    pub fn with_strategy(mut self, strategy: ResolutionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set case-insensitive matching.
    pub fn with_case_insensitive(mut self, ci: bool) -> Self {
        self.case_insensitive = ci;
        self
    }

    /// Deduplicate a dataset.
    pub fn dedup(&self, data: &[Row]) -> Result<DedupResult, DedupError> {
        if self.key_fields.is_empty() {
            return Err(DedupError::NoKeyFields);
        }

        match &self.match_mode {
            MatchMode::Exact => self.dedup_exact(data),
            MatchMode::Fuzzy { max_distance } => self.dedup_fuzzy(data, *max_distance),
        }
    }

    /// Extract the key string from a record.
    fn extract_key(&self, record: &Row) -> String {
        let mut parts = Vec::with_capacity(self.key_fields.len());
        for field in &self.key_fields {
            let val = record.get(field);
            let s = match val {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => String::new(),
            };
            if self.case_insensitive {
                parts.push(s.to_lowercase());
            } else {
                parts.push(s);
            }
        }
        parts.join("\x00")
    }

    /// Exact dedup: group by key, then resolve.
    fn dedup_exact(&self, data: &[Row]) -> Result<DedupResult, DedupError> {
        // Preserve insertion order: use Vec of (key, group_index) plus a Vec of groups.
        let mut group_order: Vec<String> = Vec::new();
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();

        for (i, record) in data.iter().enumerate() {
            let key = self.extract_key(record);
            let entry = groups.entry(key.clone()).or_insert_with(|| {
                group_order.push(key.clone());
                Vec::new()
            });
            entry.push(i);
        }

        let mut records = Vec::new();
        let mut duplicates = Vec::new();
        let mut group_sizes = Vec::new();

        for key in &group_order {
            let indices = groups.get(key).unwrap();
            group_sizes.push(indices.len());

            let resolved = self.resolve_group(data, indices);
            records.push(resolved);

            // All non-kept records are duplicates.
            if indices.len() > 1 {
                for &idx in indices {
                    // The resolved record came from the group; add remaining as duplicates.
                    duplicates.push(data[idx].clone());
                }
                // Remove the last pushed duplicate (the one we kept).
                // Actually, we kept a merged/first/last, so we mark all originals as dupes
                // and the resolved one is in 'records'. But for stats, we track how many
                // were removed.
            }
        }

        // Recalculate duplicates more precisely.
        let total_dupes: usize = group_sizes.iter().map(|s| s.saturating_sub(1)).sum();
        duplicates.clear();
        for key in &group_order {
            let indices = groups.get(key).unwrap();
            if indices.len() > 1 {
                // The indices that are NOT kept.
                let skip_idx = match self.strategy {
                    ResolutionStrategy::KeepFirst | ResolutionStrategy::MergeFirst => indices[0],
                    ResolutionStrategy::KeepLast | ResolutionStrategy::MergeLast => {
                        *indices.last().unwrap()
                    }
                };
                for &idx in indices {
                    if idx != skip_idx {
                        duplicates.push(data[idx].clone());
                    }
                }
            }
        }

        let _ = total_dupes; // used implicitly via stats
        let stats = DedupStats::compute(data.len(), records.len(), &group_sizes);

        Ok(DedupResult {
            records,
            duplicates,
            stats,
        })
    }

    /// Fuzzy dedup using edit distance.
    fn dedup_fuzzy(
        &self,
        data: &[Row],
        max_distance: usize,
    ) -> Result<DedupResult, DedupError> {
        let keys: Vec<String> = data.iter().map(|r| self.extract_key(r)).collect();
        let mut used = vec![false; data.len()];
        let mut groups: Vec<Vec<usize>> = Vec::new();

        for i in 0..data.len() {
            if used[i] {
                continue;
            }
            let mut group = vec![i];
            used[i] = true;

            for j in (i + 1)..data.len() {
                if used[j] {
                    continue;
                }
                let dist = edit_distance(&keys[i], &keys[j]);
                if dist <= max_distance {
                    group.push(j);
                    used[j] = true;
                }
            }
            groups.push(group);
        }

        let mut records = Vec::new();
        let mut duplicates = Vec::new();
        let mut group_sizes = Vec::new();

        for group in &groups {
            group_sizes.push(group.len());
            let resolved = self.resolve_group(data, group);
            records.push(resolved);

            if group.len() > 1 {
                let skip_idx = match self.strategy {
                    ResolutionStrategy::KeepFirst | ResolutionStrategy::MergeFirst => group[0],
                    ResolutionStrategy::KeepLast | ResolutionStrategy::MergeLast => {
                        *group.last().unwrap()
                    }
                };
                for &idx in group {
                    if idx != skip_idx {
                        duplicates.push(data[idx].clone());
                    }
                }
            }
        }

        let stats = DedupStats::compute(data.len(), records.len(), &group_sizes);

        Ok(DedupResult {
            records,
            duplicates,
            stats,
        })
    }

    /// Resolve a group of duplicate records into a single record.
    fn resolve_group(&self, data: &[Row], indices: &[usize]) -> Row {
        match self.strategy {
            ResolutionStrategy::KeepFirst => data[indices[0]].clone(),
            ResolutionStrategy::KeepLast => data[*indices.last().unwrap()].clone(),
            ResolutionStrategy::MergeLast => {
                let mut merged = Row::new();
                for &idx in indices {
                    for (k, v) in &data[idx] {
                        merged.insert(k.clone(), v.clone());
                    }
                }
                merged
            }
            ResolutionStrategy::MergeFirst => {
                let mut merged = Row::new();
                // Iterate in reverse so first values win.
                for &idx in indices.iter().rev() {
                    for (k, v) in &data[idx] {
                        merged.insert(k.clone(), v.clone());
                    }
                }
                merged
            }
        }
    }
}

// ── Sliding window dedup ─────────────────────────────────────────

/// Sliding window deduplication for streaming data.
pub struct SlidingWindowDedup {
    /// Key fields.
    key_fields: Vec<String>,
    /// Maximum window size.
    window_size: usize,
    /// Window entries.
    window: Vec<WindowEntry>,
    /// Next sequence number.
    next_seq: u64,
    /// Total records seen.
    total_seen: u64,
    /// Total duplicates skipped.
    total_dupes: u64,
    /// Case-insensitive matching.
    case_insensitive: bool,
}

impl SlidingWindowDedup {
    /// Create a new sliding window dedup.
    pub fn new(key_fields: Vec<String>, window_size: usize) -> Self {
        Self {
            key_fields,
            window_size,
            window: Vec::new(),
            next_seq: 0,
            total_seen: 0,
            total_dupes: 0,
            case_insensitive: false,
        }
    }

    /// Set case-insensitive matching.
    pub fn with_case_insensitive(mut self, ci: bool) -> Self {
        self.case_insensitive = ci;
        self
    }

    /// Process a single record. Returns `Some(record)` if not a duplicate.
    pub fn process(&mut self, record: Row) -> Option<Row> {
        self.total_seen += 1;
        let key = self.extract_key(&record);

        // Check if key is in window.
        let is_dup = self.window.iter().any(|e| e.key == key);

        if is_dup {
            self.total_dupes += 1;
            return None;
        }

        // Add to window.
        let seq = self.next_seq;
        self.next_seq += 1;
        self.window.push(WindowEntry {
            key,
            record: record.clone(),
            sequence: seq,
        });

        // Evict oldest if over window size.
        while self.window.len() > self.window_size {
            self.window.remove(0);
        }

        Some(record)
    }

    /// Get current window size.
    pub fn current_window_size(&self) -> usize {
        self.window.len()
    }

    /// Get dedup statistics.
    pub fn stats(&self) -> DedupStats {
        let output = self.total_seen - self.total_dupes;
        DedupStats {
            total_input: self.total_seen as usize,
            total_output: output as usize,
            duplicates_found: self.total_dupes as usize,
            duplicate_groups: 0,
            largest_group: 0,
            dedup_ratio: if self.total_seen == 0 {
                0.0
            } else {
                self.total_dupes as f64 / self.total_seen as f64
            },
        }
    }

    /// Clear the window.
    pub fn clear(&mut self) {
        self.window.clear();
    }

    fn extract_key(&self, record: &Row) -> String {
        let mut parts = Vec::new();
        for field in &self.key_fields {
            let val = record.get(field);
            let s = match val {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => String::new(),
            };
            if self.case_insensitive {
                parts.push(s.to_lowercase());
            } else {
                parts.push(s);
            }
        }
        parts.join("\x00")
    }
}

impl fmt::Debug for SlidingWindowDedup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlidingWindowDedup")
            .field("key_fields", &self.key_fields)
            .field("window_size", &self.window_size)
            .field("current_size", &self.window.len())
            .field("total_seen", &self.total_seen)
            .field("total_dupes", &self.total_dupes)
            .finish()
    }
}

// ── Edit distance ────────────────────────────────────────────────

/// Compute the Levenshtein edit distance between two strings.
fn edit_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for j in 0..=n {
        prev[j] = j;
    }

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pairs: &[(&str, serde_json::Value)]) -> Row {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn no_key_fields_error() {
        let engine = DedupEngine::new(vec![]);
        let err = engine.dedup(&[]).unwrap_err();
        assert_eq!(err, DedupError::NoKeyFields);
    }

    #[test]
    fn no_duplicates() {
        let engine = DedupEngine::new(vec!["id".into()]);
        let data = vec![
            row(&[("id", serde_json::json!("a"))]),
            row(&[("id", serde_json::json!("b"))]),
            row(&[("id", serde_json::json!("c"))]),
        ];
        let result = engine.dedup(&data).unwrap();
        assert_eq!(result.records.len(), 3);
        assert_eq!(result.stats.duplicates_found, 0);
    }

    #[test]
    fn exact_dedup_keep_first() {
        let engine = DedupEngine::new(vec!["name".into()])
            .with_strategy(ResolutionStrategy::KeepFirst);

        let data = vec![
            row(&[("name", serde_json::json!("Alice")), ("age", serde_json::json!(30))]),
            row(&[("name", serde_json::json!("Bob")), ("age", serde_json::json!(25))]),
            row(&[("name", serde_json::json!("Alice")), ("age", serde_json::json!(31))]),
        ];
        let result = engine.dedup(&data).unwrap();
        assert_eq!(result.records.len(), 2);
        // First Alice should be kept.
        let alice = result.records.iter().find(|r| {
            r.get("name") == Some(&serde_json::json!("Alice"))
        }).unwrap();
        assert_eq!(alice.get("age"), Some(&serde_json::json!(30)));
    }

    #[test]
    fn exact_dedup_keep_last() {
        let engine = DedupEngine::new(vec!["name".into()])
            .with_strategy(ResolutionStrategy::KeepLast);

        let data = vec![
            row(&[("name", serde_json::json!("Alice")), ("age", serde_json::json!(30))]),
            row(&[("name", serde_json::json!("Alice")), ("age", serde_json::json!(31))]),
        ];
        let result = engine.dedup(&data).unwrap();
        assert_eq!(result.records.len(), 1);
        assert_eq!(result.records[0].get("age"), Some(&serde_json::json!(31)));
    }

    #[test]
    fn exact_dedup_merge_last() {
        let engine = DedupEngine::new(vec!["id".into()])
            .with_strategy(ResolutionStrategy::MergeLast);

        let data = vec![
            row(&[("id", serde_json::json!(1)), ("a", serde_json::json!("x"))]),
            row(&[("id", serde_json::json!(1)), ("b", serde_json::json!("y"))]),
        ];
        let result = engine.dedup(&data).unwrap();
        assert_eq!(result.records.len(), 1);
        let rec = &result.records[0];
        // Both fields should be present (merged).
        assert!(rec.contains_key("a"));
        assert!(rec.contains_key("b"));
    }

    #[test]
    fn exact_dedup_merge_first() {
        let engine = DedupEngine::new(vec!["id".into()])
            .with_strategy(ResolutionStrategy::MergeFirst);

        let data = vec![
            row(&[("id", serde_json::json!(1)), ("val", serde_json::json!("first"))]),
            row(&[("id", serde_json::json!(1)), ("val", serde_json::json!("second"))]),
        ];
        let result = engine.dedup(&data).unwrap();
        assert_eq!(result.records.len(), 1);
        // MergeFirst: first value wins for conflicts.
        assert_eq!(result.records[0].get("val"), Some(&serde_json::json!("first")));
    }

    #[test]
    fn composite_key_dedup() {
        let engine = DedupEngine::new(vec!["first".into(), "last".into()]);

        let data = vec![
            row(&[("first", serde_json::json!("Alice")), ("last", serde_json::json!("Smith"))]),
            row(&[("first", serde_json::json!("Alice")), ("last", serde_json::json!("Jones"))]),
            row(&[("first", serde_json::json!("Alice")), ("last", serde_json::json!("Smith"))]),
        ];
        let result = engine.dedup(&data).unwrap();
        assert_eq!(result.records.len(), 2);
        assert_eq!(result.stats.duplicates_found, 1);
    }

    #[test]
    fn case_insensitive_dedup() {
        let engine = DedupEngine::new(vec!["name".into()])
            .with_case_insensitive(true);

        let data = vec![
            row(&[("name", serde_json::json!("Alice"))]),
            row(&[("name", serde_json::json!("alice"))]),
            row(&[("name", serde_json::json!("ALICE"))]),
        ];
        let result = engine.dedup(&data).unwrap();
        assert_eq!(result.records.len(), 1);
        assert_eq!(result.stats.duplicates_found, 2);
    }

    #[test]
    fn fuzzy_dedup() {
        let engine = DedupEngine::new(vec!["name".into()])
            .with_match_mode(MatchMode::Fuzzy { max_distance: 1 });

        let data = vec![
            row(&[("name", serde_json::json!("Alice"))]),
            row(&[("name", serde_json::json!("Alce"))]),   // distance 1 from Alice
            row(&[("name", serde_json::json!("Bob"))]),     // different
        ];
        let result = engine.dedup(&data).unwrap();
        assert_eq!(result.records.len(), 2); // Alice+Alce merged, Bob separate
    }

    #[test]
    fn fuzzy_dedup_no_match() {
        let engine = DedupEngine::new(vec!["name".into()])
            .with_match_mode(MatchMode::Fuzzy { max_distance: 0 });

        let data = vec![
            row(&[("name", serde_json::json!("Alice"))]),
            row(&[("name", serde_json::json!("Bob"))]),
        ];
        let result = engine.dedup(&data).unwrap();
        assert_eq!(result.records.len(), 2);
    }

    #[test]
    fn dedup_stats() {
        let engine = DedupEngine::new(vec!["id".into()]);

        let data = vec![
            row(&[("id", serde_json::json!(1))]),
            row(&[("id", serde_json::json!(2))]),
            row(&[("id", serde_json::json!(1))]),
            row(&[("id", serde_json::json!(1))]),
            row(&[("id", serde_json::json!(3))]),
        ];
        let result = engine.dedup(&data).unwrap();
        assert_eq!(result.stats.total_input, 5);
        assert_eq!(result.stats.total_output, 3);
        assert_eq!(result.stats.duplicates_found, 2);
        assert_eq!(result.stats.duplicate_groups, 1);
        assert_eq!(result.stats.largest_group, 3);
    }

    #[test]
    fn dedup_ratio() {
        let stats = DedupStats::compute(10, 7, &[3, 1, 1, 2, 1, 1, 1]);
        assert!((stats.dedup_ratio - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn duplicates_list() {
        let engine = DedupEngine::new(vec!["id".into()]);

        let data = vec![
            row(&[("id", serde_json::json!(1)), ("v", serde_json::json!("a"))]),
            row(&[("id", serde_json::json!(1)), ("v", serde_json::json!("b"))]),
        ];
        let result = engine.dedup(&data).unwrap();
        assert_eq!(result.duplicates.len(), 1);
    }

    #[test]
    fn sliding_window_dedup() {
        let mut sw = SlidingWindowDedup::new(vec!["id".into()], 5);

        let r1 = row(&[("id", serde_json::json!(1))]);
        let r2 = row(&[("id", serde_json::json!(2))]);
        let r1_dup = row(&[("id", serde_json::json!(1))]);
        let r3 = row(&[("id", serde_json::json!(3))]);

        assert!(sw.process(r1).is_some());
        assert!(sw.process(r2).is_some());
        assert!(sw.process(r1_dup).is_none()); // duplicate
        assert!(sw.process(r3).is_some());

        let stats = sw.stats();
        assert_eq!(stats.total_input, 4);
        assert_eq!(stats.total_output, 3);
        assert_eq!(stats.duplicates_found, 1);
    }

    #[test]
    fn sliding_window_eviction() {
        let mut sw = SlidingWindowDedup::new(vec!["id".into()], 2);

        let r1 = row(&[("id", serde_json::json!(1))]);
        let r2 = row(&[("id", serde_json::json!(2))]);
        let r3 = row(&[("id", serde_json::json!(3))]);
        // After r3, r1 should be evicted from the window.
        let r1_again = row(&[("id", serde_json::json!(1))]);

        assert!(sw.process(r1).is_some());
        assert!(sw.process(r2).is_some());
        assert!(sw.process(r3).is_some());
        assert_eq!(sw.current_window_size(), 2);
        // r1 was evicted, so it's no longer a duplicate.
        assert!(sw.process(r1_again).is_some());
    }

    #[test]
    fn sliding_window_clear() {
        let mut sw = SlidingWindowDedup::new(vec!["id".into()], 10);
        sw.process(row(&[("id", serde_json::json!(1))]));
        assert_eq!(sw.current_window_size(), 1);
        sw.clear();
        assert_eq!(sw.current_window_size(), 0);
    }

    #[test]
    fn edit_distance_identical() {
        assert_eq!(edit_distance("hello", "hello"), 0);
    }

    #[test]
    fn edit_distance_one_insert() {
        assert_eq!(edit_distance("hello", "helo"), 1);
    }

    #[test]
    fn edit_distance_substitution() {
        assert_eq!(edit_distance("kitten", "sitten"), 1);
    }

    #[test]
    fn edit_distance_empty() {
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("", ""), 0);
    }

    #[test]
    fn sliding_window_case_insensitive() {
        let mut sw = SlidingWindowDedup::new(vec!["name".into()], 10)
            .with_case_insensitive(true);

        let r1 = row(&[("name", serde_json::json!("Alice"))]);
        let r2 = row(&[("name", serde_json::json!("alice"))]);

        assert!(sw.process(r1).is_some());
        assert!(sw.process(r2).is_none()); // case-insensitive duplicate
    }

    #[test]
    fn error_display() {
        let e = DedupError::NoKeyFields;
        assert!(format!("{e}").contains("no key fields"));
        let e2 = DedupError::FieldNotFound {
            record_index: 3,
            field: "x".into(),
        };
        assert!(format!("{e2}").contains("field"));
    }

    #[test]
    fn empty_dataset() {
        let engine = DedupEngine::new(vec!["id".into()]);
        let result = engine.dedup(&[]).unwrap();
        assert_eq!(result.records.len(), 0);
        assert_eq!(result.stats.total_input, 0);
    }

    #[test]
    fn missing_key_field_uses_empty() {
        let engine = DedupEngine::new(vec!["id".into()]);
        let data = vec![
            row(&[("name", serde_json::json!("Alice"))]),
            row(&[("name", serde_json::json!("Bob"))]),
        ];
        // Both records have empty key for "id", so they are duplicates.
        let result = engine.dedup(&data).unwrap();
        assert_eq!(result.records.len(), 1);
    }
}
