//! Snapshot testing — value serialization, snapshot comparison, update mode,
//! inline snapshots, diff display, and snapshot file management.
//!
//! Replaces JS snapshot testing libraries (Jest snapshots, snap-shot-it) with a
//! pure-Rust snapshot testing framework that supports both file-based and inline
//! snapshot workflows.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Snapshot testing errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotError {
    /// Snapshot mismatch — expected vs actual differ.
    Mismatch { name: String, expected: String, actual: String },
    /// Snapshot not found (first run).
    NotFound(String),
    /// Duplicate snapshot name within the same suite.
    DuplicateName(String),
    /// Serialization error.
    SerializationError(String),
    /// Invalid snapshot file format.
    InvalidFormat(String),
    /// Suite not found.
    SuiteNotFound(String),
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch { name, expected, actual } => {
                write!(f, "snapshot mismatch for '{name}':\n  expected: {expected}\n  actual:   {actual}")
            }
            Self::NotFound(name) => write!(f, "snapshot not found: {name}"),
            Self::DuplicateName(name) => write!(f, "duplicate snapshot name: {name}"),
            Self::SerializationError(msg) => write!(f, "serialization error: {msg}"),
            Self::InvalidFormat(msg) => write!(f, "invalid snapshot format: {msg}"),
            Self::SuiteNotFound(name) => write!(f, "suite not found: {name}"),
        }
    }
}

impl std::error::Error for SnapshotError {}

// ── Update Mode ────────────────────────────────────────────────

/// Controls how snapshot mismatches are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateMode {
    /// Fail on mismatch (default for CI).
    Strict,
    /// Prompt the user to update (interactive).
    Interactive,
    /// Automatically accept new snapshots.
    AutoAccept,
}

impl Default for UpdateMode {
    fn default() -> Self {
        Self::Strict
    }
}

// ── Diff Display ───────────────────────────────────────────────

/// A single line in a diff output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffLine {
    /// Line present in both sides.
    Context(String),
    /// Line only in expected (removed).
    Removed(String),
    /// Line only in actual (added).
    Added(String),
}

impl fmt::Display for DiffLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Context(line) => write!(f, "  {line}"),
            Self::Removed(line) => write!(f, "- {line}"),
            Self::Added(line) => write!(f, "+ {line}"),
        }
    }
}

/// Compute a simple line-based diff between two strings.
pub fn compute_diff(expected: &str, actual: &str) -> Vec<DiffLine> {
    let exp_lines: Vec<&str> = expected.lines().collect();
    let act_lines: Vec<&str> = actual.lines().collect();

    // LCS-based diff
    let m = exp_lines.len();
    let n = act_lines.len();
    let mut lcs = vec![vec![0u32; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if exp_lines[i - 1] == act_lines[j - 1] {
                lcs[i][j] = lcs[i - 1][j - 1] + 1;
            } else {
                lcs[i][j] = lcs[i - 1][j].max(lcs[i][j - 1]);
            }
        }
    }

    let mut result = Vec::new();
    let mut i = m;
    let mut j = n;

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && exp_lines[i - 1] == act_lines[j - 1] {
            result.push(DiffLine::Context(exp_lines[i - 1].to_string()));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || lcs[i][j - 1] >= lcs[i - 1][j]) {
            result.push(DiffLine::Added(act_lines[j - 1].to_string()));
            j -= 1;
        } else {
            result.push(DiffLine::Removed(exp_lines[i - 1].to_string()));
            i -= 1;
        }
    }

    result.reverse();
    result
}

/// Format a diff as a human-readable string.
pub fn format_diff(diff: &[DiffLine]) -> String {
    diff.iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

// ── Snapshot Entry ─────────────────────────────────────────────

/// A single stored snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotEntry {
    /// Snapshot name (unique within suite).
    pub name: String,
    /// Serialized value.
    pub value: String,
    /// Number of times this snapshot has been verified.
    pub hit_count: u64,
    /// Whether snapshot was updated in this run.
    pub updated: bool,
}

/// Result of comparing a value against a stored snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotResult {
    /// Snapshot matched.
    Match,
    /// New snapshot created (first time).
    Created,
    /// Snapshot was updated.
    Updated { old: String, new: String },
    /// Snapshot mismatch (strict mode).
    Mismatch { diff: Vec<DiffLine> },
}

// ── Inline Snapshot ────────────────────────────────────────────

/// An inline snapshot stored directly in code (simulated).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineSnapshot {
    /// File path where the inline snapshot lives.
    pub file: String,
    /// Line number in the file.
    pub line: u32,
    /// The expected value (embedded in source).
    pub expected: String,
}

// ── Snapshot Serializer ────────────────────────────────────────

/// Serializer for snapshot values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapshotSerializer {
    /// Pretty-printed JSON (default).
    Json,
    /// Debug format.
    Debug,
    /// Display format.
    Display,
    /// Raw string (no transformation).
    Raw,
}

impl Default for SnapshotSerializer {
    fn default() -> Self {
        Self::Json
    }
}

/// Serialize a JSON value to a snapshot string.
pub fn serialize_value(value: &serde_json::Value, serializer: SnapshotSerializer) -> String {
    match serializer {
        SnapshotSerializer::Json => {
            serde_json::to_string_pretty(value).unwrap_or_else(|_| format!("{value:?}"))
        }
        SnapshotSerializer::Debug => format!("{value:?}"),
        SnapshotSerializer::Display => format!("{value}"),
        SnapshotSerializer::Raw => value.to_string(),
    }
}

// ── Snapshot File ──────────────────────────────────────────────

/// A snapshot file containing multiple entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotFile {
    /// File path (virtual — no actual I/O).
    pub path: String,
    /// Entries keyed by snapshot name.
    entries: HashMap<String, SnapshotEntry>,
}

impl SnapshotFile {
    /// Create a new snapshot file.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            entries: HashMap::new(),
        }
    }

    /// Get a snapshot entry by name.
    pub fn get(&self, name: &str) -> Option<&SnapshotEntry> {
        self.entries.get(name)
    }

    /// Set or update a snapshot entry.
    pub fn set(&mut self, name: impl Into<String>, value: impl Into<String>) {
        let name = name.into();
        let value = value.into();
        let entry = self.entries.entry(name.clone()).or_insert(SnapshotEntry {
            name: name.clone(),
            value: String::new(),
            hit_count: 0,
            updated: false,
        });
        if entry.value != value {
            entry.updated = true;
        }
        entry.value = value;
        entry.hit_count += 1;
    }

    /// Remove a snapshot entry.
    pub fn remove(&mut self, name: &str) -> Option<SnapshotEntry> {
        self.entries.remove(name)
    }

    /// List all snapshot names (sorted).
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.entries.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of snapshots in this file.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if file has no snapshots.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries that were not hit in this run (obsolete snapshots).
    pub fn prune_obsolete(&mut self) -> Vec<String> {
        let obsolete: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.hit_count == 0)
            .map(|(k, _)| k.clone())
            .collect();
        for name in &obsolete {
            self.entries.remove(name);
        }
        let mut sorted = obsolete;
        sorted.sort();
        sorted
    }

    /// Serialize the snapshot file to a string representation.
    pub fn serialize(&self) -> String {
        let mut out = String::new();
        let mut names = self.names();
        names.sort();
        for name in &names {
            if let Some(entry) = self.entries.get(name) {
                out.push_str(&format!("--- {} ---\n", entry.name));
                out.push_str(&entry.value);
                out.push_str("\n\n");
            }
        }
        out
    }

    /// Parse a snapshot file from its string representation.
    pub fn parse(path: impl Into<String>, content: &str) -> Result<Self, SnapshotError> {
        let path = path.into();
        let mut file = Self::new(path);
        let mut current_name: Option<String> = None;
        let mut current_value = String::new();

        for line in content.lines() {
            if let Some(stripped) = line.strip_prefix("--- ") {
                if let Some(name_part) = stripped.strip_suffix(" ---") {
                    // Save previous entry
                    if let Some(name) = current_name.take() {
                        let trimmed = current_value.trim_end().to_string();
                        file.set(name, trimmed);
                        current_value.clear();
                    }
                    current_name = Some(name_part.to_string());
                    continue;
                }
            }
            if current_name.is_some() {
                if !current_value.is_empty() {
                    current_value.push('\n');
                }
                current_value.push_str(line);
            }
        }

        // Save last entry
        if let Some(name) = current_name {
            let trimmed = current_value.trim_end().to_string();
            file.set(name, trimmed);
        }

        Ok(file)
    }
}

// ── Snapshot Suite ─────────────────────────────────────────────

/// A snapshot test suite managing multiple snapshot files.
#[derive(Debug, Clone)]
pub struct SnapshotSuite {
    /// Suite name (usually the test module name).
    pub name: String,
    /// Update mode for this suite.
    pub update_mode: UpdateMode,
    /// Serializer format.
    pub serializer: SnapshotSerializer,
    /// Snapshot files managed by this suite.
    files: HashMap<String, SnapshotFile>,
    /// Inline snapshots collected during the run.
    inline_snapshots: Vec<InlineSnapshot>,
    /// Count of assertions made.
    assertion_count: u64,
    /// Count of mismatches found.
    mismatch_count: u64,
    /// Count of new snapshots created.
    created_count: u64,
}

impl SnapshotSuite {
    /// Create a new snapshot suite.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            update_mode: UpdateMode::default(),
            serializer: SnapshotSerializer::default(),
            files: HashMap::new(),
            inline_snapshots: Vec::new(),
            assertion_count: 0,
            mismatch_count: 0,
            created_count: 0,
        }
    }

    /// Set the update mode.
    pub fn with_update_mode(mut self, mode: UpdateMode) -> Self {
        self.update_mode = mode;
        self
    }

    /// Set the serializer.
    pub fn with_serializer(mut self, serializer: SnapshotSerializer) -> Self {
        self.serializer = serializer;
        self
    }

    /// Register a snapshot file (pre-existing snapshots).
    pub fn register_file(&mut self, file: SnapshotFile) {
        self.files.insert(file.path.clone(), file);
    }

    /// Assert a value against a named snapshot in a file.
    pub fn assert_snapshot(
        &mut self,
        file_path: &str,
        snapshot_name: &str,
        actual_value: &serde_json::Value,
    ) -> Result<SnapshotResult, SnapshotError> {
        self.assertion_count += 1;
        let actual_str = serialize_value(actual_value, self.serializer);

        let file = self
            .files
            .entry(file_path.to_string())
            .or_insert_with(|| SnapshotFile::new(file_path));

        if let Some(existing) = file.get(snapshot_name) {
            if existing.value == actual_str {
                file.set(snapshot_name, actual_str);
                Ok(SnapshotResult::Match)
            } else {
                let diff = compute_diff(&existing.value, &actual_str);
                match self.update_mode {
                    UpdateMode::Strict => {
                        self.mismatch_count += 1;
                        Ok(SnapshotResult::Mismatch { diff })
                    }
                    UpdateMode::AutoAccept | UpdateMode::Interactive => {
                        let old = existing.value.clone();
                        file.set(snapshot_name, actual_str.clone());
                        Ok(SnapshotResult::Updated { old, new: actual_str })
                    }
                }
            }
        } else {
            // New snapshot
            self.created_count += 1;
            file.set(snapshot_name, actual_str);
            Ok(SnapshotResult::Created)
        }
    }

    /// Assert an inline snapshot (compare directly with expected string).
    pub fn assert_inline(
        &mut self,
        file: &str,
        line: u32,
        expected: &str,
        actual_value: &serde_json::Value,
    ) -> Result<SnapshotResult, SnapshotError> {
        self.assertion_count += 1;
        let actual_str = serialize_value(actual_value, self.serializer);

        self.inline_snapshots.push(InlineSnapshot {
            file: file.to_string(),
            line,
            expected: expected.to_string(),
        });

        if expected == actual_str {
            Ok(SnapshotResult::Match)
        } else {
            let diff = compute_diff(expected, &actual_str);
            match self.update_mode {
                UpdateMode::Strict => {
                    self.mismatch_count += 1;
                    Ok(SnapshotResult::Mismatch { diff })
                }
                UpdateMode::AutoAccept | UpdateMode::Interactive => {
                    Ok(SnapshotResult::Updated {
                        old: expected.to_string(),
                        new: actual_str,
                    })
                }
            }
        }
    }

    /// Get statistics for the current run.
    pub fn stats(&self) -> SnapshotStats {
        SnapshotStats {
            assertions: self.assertion_count,
            mismatches: self.mismatch_count,
            created: self.created_count,
            files: self.files.len() as u64,
            total_snapshots: self.files.values().map(|f| f.len() as u64).sum(),
        }
    }

    /// Get the snapshot file for a given path.
    pub fn file(&self, path: &str) -> Option<&SnapshotFile> {
        self.files.get(path)
    }

    /// Prune obsolete snapshots from all files.
    pub fn prune_all(&mut self) -> Vec<(String, Vec<String>)> {
        let mut result = Vec::new();
        for (path, file) in &mut self.files {
            let pruned = file.prune_obsolete();
            if !pruned.is_empty() {
                result.push((path.clone(), pruned));
            }
        }
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}

/// Statistics from a snapshot run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotStats {
    /// Total assertions made.
    pub assertions: u64,
    /// Total mismatches.
    pub mismatches: u64,
    /// New snapshots created.
    pub created: u64,
    /// Number of snapshot files.
    pub files: u64,
    /// Total snapshots across all files.
    pub total_snapshots: u64,
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_diff_identical() {
        let diff = compute_diff("hello\nworld", "hello\nworld");
        assert_eq!(diff.len(), 2);
        assert!(diff.iter().all(|d| matches!(d, DiffLine::Context(_))));
    }

    #[test]
    fn test_diff_added_line() {
        let diff = compute_diff("hello", "hello\nworld");
        let has_added = diff.iter().any(|d| matches!(d, DiffLine::Added(s) if s == "world"));
        assert!(has_added);
    }

    #[test]
    fn test_diff_removed_line() {
        let diff = compute_diff("hello\nworld", "hello");
        let has_removed = diff.iter().any(|d| matches!(d, DiffLine::Removed(s) if s == "world"));
        assert!(has_removed);
    }

    #[test]
    fn test_diff_changed_line() {
        let diff = compute_diff("hello\nfoo", "hello\nbar");
        let has_removed = diff.iter().any(|d| matches!(d, DiffLine::Removed(s) if s == "foo"));
        let has_added = diff.iter().any(|d| matches!(d, DiffLine::Added(s) if s == "bar"));
        assert!(has_removed);
        assert!(has_added);
    }

    #[test]
    fn test_diff_empty_to_content() {
        let diff = compute_diff("", "line1\nline2");
        let added_count = diff.iter().filter(|d| matches!(d, DiffLine::Added(_))).count();
        assert_eq!(added_count, 2);
    }

    #[test]
    fn test_format_diff() {
        let diff = vec![
            DiffLine::Context("same".to_string()),
            DiffLine::Removed("old".to_string()),
            DiffLine::Added("new".to_string()),
        ];
        let formatted = format_diff(&diff);
        assert!(formatted.contains("  same"));
        assert!(formatted.contains("- old"));
        assert!(formatted.contains("+ new"));
    }

    #[test]
    fn test_serialize_json() {
        let val = json!({"name": "test", "count": 42});
        let out = serialize_value(&val, SnapshotSerializer::Json);
        assert!(out.contains("\"name\": \"test\""));
        assert!(out.contains("\"count\": 42"));
    }

    #[test]
    fn test_serialize_debug() {
        let val = json!("hello");
        let out = serialize_value(&val, SnapshotSerializer::Debug);
        assert!(out.contains("hello"));
    }

    #[test]
    fn test_serialize_display() {
        let val = json!(42);
        let out = serialize_value(&val, SnapshotSerializer::Display);
        assert_eq!(out, "42");
    }

    #[test]
    fn test_serialize_raw() {
        let val = json!({"a": 1});
        let out = serialize_value(&val, SnapshotSerializer::Raw);
        assert!(out.contains("\"a\":1") || out.contains("\"a\": 1"));
    }

    #[test]
    fn test_snapshot_file_crud() {
        let mut file = SnapshotFile::new("test.snap");
        assert!(file.is_empty());
        file.set("snap1", "value1");
        assert_eq!(file.len(), 1);
        assert_eq!(file.get("snap1").unwrap().value, "value1");
        file.set("snap1", "value2");
        assert_eq!(file.get("snap1").unwrap().value, "value2");
        assert!(file.get("snap1").unwrap().updated);
        file.remove("snap1");
        assert!(file.is_empty());
    }

    #[test]
    fn test_snapshot_file_names_sorted() {
        let mut file = SnapshotFile::new("test.snap");
        file.set("charlie", "c");
        file.set("alpha", "a");
        file.set("bravo", "b");
        assert_eq!(file.names(), vec!["alpha", "bravo", "charlie"]);
    }

    #[test]
    fn test_snapshot_file_prune_obsolete() {
        let mut file = SnapshotFile::new("test.snap");
        file.set("active", "val1");
        // Insert a stale entry with zero hit_count
        file.entries.insert(
            "stale".to_string(),
            SnapshotEntry {
                name: "stale".to_string(),
                value: "old".to_string(),
                hit_count: 0,
                updated: false,
            },
        );
        assert_eq!(file.len(), 2);
        let pruned = file.prune_obsolete();
        assert_eq!(pruned, vec!["stale"]);
        assert_eq!(file.len(), 1);
    }

    #[test]
    fn test_snapshot_file_serialize_parse_roundtrip() {
        let mut file = SnapshotFile::new("roundtrip.snap");
        file.set("first", "value one");
        file.set("second", "value two\nwith newlines");

        let serialized = file.serialize();
        let parsed = SnapshotFile::parse("roundtrip.snap", &serialized).unwrap();

        assert_eq!(parsed.get("first").unwrap().value, "value one");
        assert_eq!(parsed.get("second").unwrap().value, "value two\nwith newlines");
    }

    #[test]
    fn test_suite_new_snapshot_created() {
        let mut suite = SnapshotSuite::new("my_tests");
        let result = suite
            .assert_snapshot("test.snap", "greeting", &json!("hello"))
            .unwrap();
        assert_eq!(result, SnapshotResult::Created);
        assert_eq!(suite.stats().created, 1);
    }

    #[test]
    fn test_suite_snapshot_match() {
        let mut suite = SnapshotSuite::new("my_tests");
        suite.assert_snapshot("test.snap", "val", &json!(42)).unwrap();
        let result = suite.assert_snapshot("test.snap", "val", &json!(42)).unwrap();
        assert_eq!(result, SnapshotResult::Match);
    }

    #[test]
    fn test_suite_snapshot_mismatch_strict() {
        let mut suite = SnapshotSuite::new("my_tests")
            .with_update_mode(UpdateMode::Strict);
        suite.assert_snapshot("test.snap", "val", &json!(1)).unwrap();
        let result = suite.assert_snapshot("test.snap", "val", &json!(2)).unwrap();
        assert!(matches!(result, SnapshotResult::Mismatch { .. }));
        assert_eq!(suite.stats().mismatches, 1);
    }

    #[test]
    fn test_suite_snapshot_auto_update() {
        let mut suite = SnapshotSuite::new("my_tests")
            .with_update_mode(UpdateMode::AutoAccept);
        suite.assert_snapshot("test.snap", "val", &json!("old")).unwrap();
        let result = suite.assert_snapshot("test.snap", "val", &json!("new")).unwrap();
        assert!(matches!(result, SnapshotResult::Updated { .. }));
    }

    #[test]
    fn test_suite_inline_match() {
        let mut suite = SnapshotSuite::new("inline_tests");
        let result = suite.assert_inline("test.rs", 10, "42", &json!(42)).unwrap();
        assert_eq!(result, SnapshotResult::Match);
    }

    #[test]
    fn test_suite_inline_mismatch() {
        let mut suite = SnapshotSuite::new("inline_tests")
            .with_update_mode(UpdateMode::Strict);
        let result = suite.assert_inline("test.rs", 10, "42", &json!(99)).unwrap();
        assert!(matches!(result, SnapshotResult::Mismatch { .. }));
    }

    #[test]
    fn test_suite_stats() {
        let mut suite = SnapshotSuite::new("stats_test");
        suite.assert_snapshot("a.snap", "s1", &json!(1)).unwrap();
        suite.assert_snapshot("b.snap", "s2", &json!(2)).unwrap();
        suite.assert_snapshot("a.snap", "s3", &json!(3)).unwrap();
        let stats = suite.stats();
        assert_eq!(stats.assertions, 3);
        assert_eq!(stats.files, 2);
        assert_eq!(stats.total_snapshots, 3);
    }

    #[test]
    fn test_suite_prune_all() {
        let mut suite = SnapshotSuite::new("prune_test");
        let mut file = SnapshotFile::new("test.snap");
        file.entries.insert(
            "orphan".to_string(),
            SnapshotEntry {
                name: "orphan".to_string(),
                value: "old".to_string(),
                hit_count: 0,
                updated: false,
            },
        );
        file.set("alive", "val");
        suite.register_file(file);
        let pruned = suite.prune_all();
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0].1, vec!["orphan"]);
    }

    #[test]
    fn test_suite_with_serializer() {
        let mut suite = SnapshotSuite::new("raw_test")
            .with_serializer(SnapshotSerializer::Display);
        let result = suite.assert_snapshot("test.snap", "num", &json!(42)).unwrap();
        assert_eq!(result, SnapshotResult::Created);
        let file = suite.file("test.snap").unwrap();
        assert_eq!(file.get("num").unwrap().value, "42");
    }

    #[test]
    fn test_update_mode_default() {
        assert_eq!(UpdateMode::default(), UpdateMode::Strict);
    }

    #[test]
    fn test_serializer_default() {
        assert_eq!(SnapshotSerializer::default(), SnapshotSerializer::Json);
    }

    #[test]
    fn test_error_display() {
        let err = SnapshotError::NotFound("test".to_string());
        assert_eq!(format!("{err}"), "snapshot not found: test");

        let err = SnapshotError::DuplicateName("dup".to_string());
        assert!(format!("{err}").contains("duplicate"));
    }

    #[test]
    fn test_diff_line_display() {
        assert_eq!(format!("{}", DiffLine::Context("x".into())), "  x");
        assert_eq!(format!("{}", DiffLine::Removed("x".into())), "- x");
        assert_eq!(format!("{}", DiffLine::Added("x".into())), "+ x");
    }

    #[test]
    fn test_snapshot_file_get_nonexistent() {
        let file = SnapshotFile::new("empty.snap");
        assert!(file.get("nope").is_none());
    }

    #[test]
    fn test_parse_empty_content() {
        let file = SnapshotFile::parse("test.snap", "").unwrap();
        assert!(file.is_empty());
    }

    #[test]
    fn test_multiple_assertions_same_snapshot() {
        let mut suite = SnapshotSuite::new("multi");
        suite.assert_snapshot("f.snap", "s", &json!(1)).unwrap();
        let result = suite.assert_snapshot("f.snap", "s", &json!(1)).unwrap();
        assert_eq!(result, SnapshotResult::Match);
        let file = suite.file("f.snap").unwrap();
        assert_eq!(file.get("s").unwrap().hit_count, 2);
    }

    #[test]
    fn test_diff_multiline() {
        let old = "line1\nline2\nline3";
        let new = "line1\nmodified\nline3\nline4";
        let diff = compute_diff(old, new);
        let has_context = diff.iter().any(|d| matches!(d, DiffLine::Context(s) if s == "line1"));
        assert!(has_context);
    }
}
