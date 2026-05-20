//! Log aggregation: structured log ingestion, field extraction, log levels,
//! timestamp parsing, pattern counting, top-K values, log stream with filters,
//! and rotation/retention policies.

use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use std::collections::HashMap;

// ── Types ──

/// Log severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Fatal => "FATAL",
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "TRACE" => Some(LogLevel::Trace),
            "DEBUG" => Some(LogLevel::Debug),
            "INFO" | "INFORMATION" => Some(LogLevel::Info),
            "WARN" | "WARNING" => Some(LogLevel::Warn),
            "ERROR" | "ERR" => Some(LogLevel::Error),
            "FATAL" | "CRITICAL" | "CRIT" => Some(LogLevel::Fatal),
            _ => None,
        }
    }

    pub fn numeric_value(&self) -> u8 {
        match self {
            LogLevel::Trace => 0,
            LogLevel::Debug => 1,
            LogLevel::Info => 2,
            LogLevel::Warn => 3,
            LogLevel::Error => 4,
            LogLevel::Fatal => 5,
        }
    }
}

/// A structured log entry.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub message: String,
    pub source: Option<String>,
    pub fields: HashMap<String, String>,
    pub raw: Option<String>,
}

impl LogEntry {
    pub fn new(level: LogLevel, message: &str) -> Self {
        Self {
            timestamp: Utc::now(),
            level,
            message: message.to_string(),
            source: None,
            fields: HashMap::new(),
            raw: None,
        }
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }

    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.fields.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_timestamp(mut self, ts: DateTime<Utc>) -> Self {
        self.timestamp = ts;
        self
    }

    pub fn with_raw(mut self, raw: &str) -> Self {
        self.raw = Some(raw.to_string());
        self
    }

    pub fn get_field(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(|s| s.as_str())
    }
}

/// Filter for querying log entries.
#[derive(Debug, Clone)]
pub struct LogFilter {
    pub min_level: Option<LogLevel>,
    pub source: Option<String>,
    pub message_contains: Option<String>,
    pub field_match: HashMap<String, String>,
    pub from_time: Option<DateTime<Utc>>,
    pub to_time: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
}

impl LogFilter {
    pub fn new() -> Self {
        Self {
            min_level: None,
            source: None,
            message_contains: None,
            field_match: HashMap::new(),
            from_time: None,
            to_time: None,
            limit: None,
        }
    }

    pub fn with_min_level(mut self, level: LogLevel) -> Self {
        self.min_level = Some(level);
        self
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }

    pub fn with_message_contains(mut self, substr: &str) -> Self {
        self.message_contains = Some(substr.to_string());
        self
    }

    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.field_match.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_time_range(mut self, from: DateTime<Utc>, to: DateTime<Utc>) -> Self {
        self.from_time = Some(from);
        self.to_time = Some(to);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Check whether a log entry matches this filter.
    pub fn matches(&self, entry: &LogEntry) -> bool {
        if let Some(min) = self.min_level {
            if entry.level < min {
                return false;
            }
        }
        if let Some(src) = &self.source {
            match &entry.source {
                Some(entry_src) if entry_src == src => {}
                _ => return false,
            }
        }
        if let Some(substr) = &self.message_contains {
            if !entry.message.contains(substr.as_str()) {
                return false;
            }
        }
        for (key, value) in &self.field_match {
            match entry.fields.get(key) {
                Some(v) if v == value => {}
                _ => return false,
            }
        }
        if let Some(from) = self.from_time {
            if entry.timestamp < from {
                return false;
            }
        }
        if let Some(to) = self.to_time {
            if entry.timestamp > to {
                return false;
            }
        }
        true
    }
}

impl Default for LogFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Retention policy for log rotation.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    pub max_entries: Option<usize>,
    pub max_age: Option<Duration>,
    pub max_bytes: Option<usize>,
}

impl RetentionPolicy {
    pub fn new() -> Self {
        Self {
            max_entries: None,
            max_age: None,
            max_bytes: None,
        }
    }

    pub fn with_max_entries(mut self, n: usize) -> Self {
        self.max_entries = Some(n);
        self
    }

    pub fn with_max_age(mut self, age: Duration) -> Self {
        self.max_age = Some(age);
        self
    }

    pub fn with_max_bytes(mut self, bytes: usize) -> Self {
        self.max_bytes = Some(bytes);
        self
    }
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Top-K result entry.
#[derive(Debug, Clone)]
pub struct TopKEntry {
    pub value: String,
    pub count: usize,
}

/// Pattern counting result.
#[derive(Debug, Clone)]
pub struct PatternCount {
    pub pattern: String,
    pub count: usize,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
}

// ── Timestamp Parsing ──

/// Parse a timestamp string in common formats.
pub fn parse_timestamp(s: &str) -> Option<DateTime<Utc>> {
    // Try ISO 8601 with timezone
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }

    // Try ISO 8601 without timezone
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(dt.and_utc());
    }

    // Try ISO 8601 with fractional seconds
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(dt.and_utc());
    }

    // Try common log format
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(dt.and_utc());
    }

    // Try with fractional seconds and space
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f") {
        return Some(dt.and_utc());
    }

    // Try epoch seconds
    if let Ok(epoch) = s.parse::<i64>() {
        return DateTime::from_timestamp(epoch, 0);
    }

    // Try epoch milliseconds
    if let Ok(epoch_ms) = s.parse::<i64>() {
        let secs = epoch_ms / 1000;
        let nanos = ((epoch_ms % 1000) * 1_000_000) as u32;
        return DateTime::from_timestamp(secs, nanos);
    }

    None
}

/// Extract structured fields from a key=value style log line.
pub fn extract_fields(line: &str) -> HashMap<String, String> {
    let mut fields = HashMap::new();
    let mut remaining = line;

    while !remaining.is_empty() {
        // Find the next key=value pair
        let eq_pos = match remaining.find('=') {
            Some(pos) => pos,
            None => break,
        };

        // Extract key (last word before =)
        let key_part = &remaining[..eq_pos];
        let key = key_part
            .rsplit(|c: char| c.is_whitespace())
            .next()
            .unwrap_or("")
            .trim();

        if key.is_empty() {
            remaining = &remaining[eq_pos + 1..];
            continue;
        }

        // Extract value
        let after_eq = &remaining[eq_pos + 1..];
        let value = if after_eq.starts_with('"') {
            // Quoted value
            let end_quote = after_eq[1..].find('"').map(|p| p + 1);
            match end_quote {
                Some(end) => {
                    let v = &after_eq[1..end];
                    remaining = if end + 1 < after_eq.len() {
                        &after_eq[end + 1..]
                    } else {
                        ""
                    };
                    v
                }
                None => {
                    remaining = "";
                    &after_eq[1..]
                }
            }
        } else {
            // Unquoted value — ends at next whitespace
            let end = after_eq
                .find(|c: char| c.is_whitespace())
                .unwrap_or(after_eq.len());
            let v = &after_eq[..end];
            remaining = if end < after_eq.len() {
                &after_eq[end..]
            } else {
                ""
            };
            v
        };

        fields.insert(key.to_string(), value.to_string());
    }

    fields
}

// ── Log Aggregator ──

/// Log aggregation engine.
#[derive(Debug)]
pub struct LogAggregator {
    entries: Vec<LogEntry>,
    retention: RetentionPolicy,
    estimated_bytes: usize,
}

impl LogAggregator {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            retention: RetentionPolicy::new(),
            estimated_bytes: 0,
        }
    }

    pub fn with_retention(mut self, policy: RetentionPolicy) -> Self {
        self.retention = policy;
        self
    }

    /// Ingest a log entry.
    pub fn ingest(&mut self, entry: LogEntry) {
        let entry_size = entry.message.len()
            + entry.source.as_ref().map(|s| s.len()).unwrap_or(0)
            + entry
                .fields
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
            + 64; // overhead estimate
        self.estimated_bytes += entry_size;
        self.entries.push(entry);
        self.apply_retention();
    }

    /// Ingest a raw log line, extracting level and fields.
    pub fn ingest_raw(&mut self, line: &str) {
        let level = self.detect_level(line);
        let fields = extract_fields(line);
        let entry = LogEntry::new(level, line).with_raw(line);
        let mut entry = entry;
        for (k, v) in fields {
            entry.fields.insert(k, v);
        }
        self.ingest(entry);
    }

    fn detect_level(&self, line: &str) -> LogLevel {
        let upper = line.to_uppercase();
        if upper.contains("FATAL") || upper.contains("CRITICAL") {
            LogLevel::Fatal
        } else if upper.contains("ERROR") || upper.contains("ERR") {
            LogLevel::Error
        } else if upper.contains("WARN") || upper.contains("WARNING") {
            LogLevel::Warn
        } else if upper.contains("DEBUG") {
            LogLevel::Debug
        } else if upper.contains("TRACE") {
            LogLevel::Trace
        } else {
            LogLevel::Info
        }
    }

    /// Query entries with a filter.
    pub fn query(&self, filter: &LogFilter) -> Vec<&LogEntry> {
        let mut results: Vec<&LogEntry> = self
            .entries
            .iter()
            .filter(|e| filter.matches(e))
            .collect();

        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }
        results
    }

    /// Count entries by level.
    pub fn count_by_level(&self) -> HashMap<LogLevel, usize> {
        let mut counts = HashMap::new();
        for entry in &self.entries {
            *counts.entry(entry.level).or_insert(0) += 1;
        }
        counts
    }

    /// Count occurrences of a substring pattern.
    pub fn count_pattern(&self, pattern: &str) -> PatternCount {
        let matching: Vec<&LogEntry> = self
            .entries
            .iter()
            .filter(|e| e.message.contains(pattern))
            .collect();

        let count = matching.len();
        let first_seen = matching
            .first()
            .map(|e| e.timestamp)
            .unwrap_or_else(Utc::now);
        let last_seen = matching
            .last()
            .map(|e| e.timestamp)
            .unwrap_or_else(Utc::now);

        PatternCount {
            pattern: pattern.to_string(),
            count,
            first_seen,
            last_seen,
        }
    }

    /// Get the top-K values for a specific field.
    pub fn top_k_field_values(&self, field_name: &str, k: usize) -> Vec<TopKEntry> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for entry in &self.entries {
            if let Some(value) = entry.fields.get(field_name) {
                *counts.entry(value.clone()).or_insert(0) += 1;
            }
        }

        let mut pairs: Vec<(String, usize)> = counts.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(k);

        pairs
            .into_iter()
            .map(|(value, count)| TopKEntry { value, count })
            .collect()
    }

    /// Get the top-K source producers.
    pub fn top_k_sources(&self, k: usize) -> Vec<TopKEntry> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for entry in &self.entries {
            if let Some(src) = &entry.source {
                *counts.entry(src.clone()).or_insert(0) += 1;
            }
        }

        let mut pairs: Vec<(String, usize)> = counts.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(k);

        pairs
            .into_iter()
            .map(|(value, count)| TopKEntry { value, count })
            .collect()
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }

    /// Apply retention policy, removing old/excess entries.
    fn apply_retention(&mut self) {
        // Max entries
        if let Some(max) = self.retention.max_entries {
            while self.entries.len() > max {
                if let Some(removed) = self.entries.first() {
                    let size = removed.message.len() + 64;
                    self.estimated_bytes = self.estimated_bytes.saturating_sub(size);
                }
                self.entries.remove(0);
            }
        }

        // Max age
        if let Some(max_age) = self.retention.max_age {
            let cutoff = Utc::now() - max_age;
            let before_len = self.entries.len();
            self.entries.retain(|e| e.timestamp >= cutoff);
            let removed = before_len - self.entries.len();
            self.estimated_bytes = self.estimated_bytes.saturating_sub(removed * 100);
        }

        // Max bytes (approximate)
        if let Some(max_bytes) = self.retention.max_bytes {
            while self.estimated_bytes > max_bytes && !self.entries.is_empty() {
                if let Some(removed) = self.entries.first() {
                    let size = removed.message.len() + 64;
                    self.estimated_bytes = self.estimated_bytes.saturating_sub(size);
                }
                self.entries.remove(0);
            }
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.estimated_bytes = 0;
    }

    /// Get error rate over recent entries.
    pub fn error_rate(&self, window: usize) -> f64 {
        let start = self.entries.len().saturating_sub(window);
        let slice = &self.entries[start..];
        if slice.is_empty() {
            return 0.0;
        }
        let error_count = slice
            .iter()
            .filter(|e| e.level >= LogLevel::Error)
            .count();
        error_count as f64 / slice.len() as f64
    }
}

impl Default for LogAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_entry_creation() {
        let entry = LogEntry::new(LogLevel::Info, "Request processed")
            .with_source("api-server")
            .with_field("request_id", "abc-123")
            .with_field("status", "200");
        assert_eq!(entry.level, LogLevel::Info);
        assert_eq!(entry.source.as_deref(), Some("api-server"));
        assert_eq!(entry.get_field("request_id"), Some("abc-123"));
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Fatal);
    }

    #[test]
    fn test_log_level_from_str() {
        assert_eq!(LogLevel::from_str_loose("INFO"), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_str_loose("warning"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str_loose("ERR"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_str_loose("CRIT"), Some(LogLevel::Fatal));
        assert_eq!(LogLevel::from_str_loose("unknown"), None);
    }

    #[test]
    fn test_log_level_numeric() {
        assert!(LogLevel::Trace.numeric_value() < LogLevel::Fatal.numeric_value());
    }

    #[test]
    fn test_filter_by_level() {
        let filter = LogFilter::new().with_min_level(LogLevel::Warn);
        let info = LogEntry::new(LogLevel::Info, "info msg");
        let warn = LogEntry::new(LogLevel::Warn, "warn msg");
        let error = LogEntry::new(LogLevel::Error, "error msg");
        assert!(!filter.matches(&info));
        assert!(filter.matches(&warn));
        assert!(filter.matches(&error));
    }

    #[test]
    fn test_filter_by_source() {
        let filter = LogFilter::new().with_source("api");
        let api_entry = LogEntry::new(LogLevel::Info, "msg").with_source("api");
        let other_entry = LogEntry::new(LogLevel::Info, "msg").with_source("worker");
        assert!(filter.matches(&api_entry));
        assert!(!filter.matches(&other_entry));
    }

    #[test]
    fn test_filter_by_message() {
        let filter = LogFilter::new().with_message_contains("timeout");
        let match_entry = LogEntry::new(LogLevel::Error, "Connection timeout occurred");
        let no_match = LogEntry::new(LogLevel::Error, "Connection refused");
        assert!(filter.matches(&match_entry));
        assert!(!filter.matches(&no_match));
    }

    #[test]
    fn test_filter_by_field() {
        let filter = LogFilter::new().with_field("status", "500");
        let entry = LogEntry::new(LogLevel::Error, "err").with_field("status", "500");
        let other = LogEntry::new(LogLevel::Info, "ok").with_field("status", "200");
        assert!(filter.matches(&entry));
        assert!(!filter.matches(&other));
    }

    #[test]
    fn test_aggregator_ingest_and_query() {
        let mut agg = LogAggregator::new();
        agg.ingest(LogEntry::new(LogLevel::Info, "Request A"));
        agg.ingest(LogEntry::new(LogLevel::Error, "Request B failed"));
        agg.ingest(LogEntry::new(LogLevel::Info, "Request C"));
        assert_eq!(agg.len(), 3);

        let errors = agg.query(&LogFilter::new().with_min_level(LogLevel::Error));
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_aggregator_count_by_level() {
        let mut agg = LogAggregator::new();
        agg.ingest(LogEntry::new(LogLevel::Info, "a"));
        agg.ingest(LogEntry::new(LogLevel::Info, "b"));
        agg.ingest(LogEntry::new(LogLevel::Error, "c"));
        let counts = agg.count_by_level();
        assert_eq!(counts[&LogLevel::Info], 2);
        assert_eq!(counts[&LogLevel::Error], 1);
    }

    #[test]
    fn test_aggregator_pattern_count() {
        let mut agg = LogAggregator::new();
        agg.ingest(LogEntry::new(LogLevel::Error, "Connection timeout to db-1"));
        agg.ingest(LogEntry::new(LogLevel::Error, "Connection timeout to db-2"));
        agg.ingest(LogEntry::new(LogLevel::Info, "Request completed"));
        let pc = agg.count_pattern("timeout");
        assert_eq!(pc.count, 2);
    }

    #[test]
    fn test_aggregator_top_k() {
        let mut agg = LogAggregator::new();
        for _ in 0..5 {
            agg.ingest(
                LogEntry::new(LogLevel::Info, "req")
                    .with_field("endpoint", "/api/users"),
            );
        }
        for _ in 0..3 {
            agg.ingest(
                LogEntry::new(LogLevel::Info, "req")
                    .with_field("endpoint", "/api/posts"),
            );
        }
        agg.ingest(
            LogEntry::new(LogLevel::Info, "req")
                .with_field("endpoint", "/api/health"),
        );
        let top = agg.top_k_field_values("endpoint", 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].value, "/api/users");
        assert_eq!(top[0].count, 5);
    }

    #[test]
    fn test_aggregator_top_k_sources() {
        let mut agg = LogAggregator::new();
        for _ in 0..4 {
            agg.ingest(LogEntry::new(LogLevel::Info, "msg").with_source("api"));
        }
        for _ in 0..2 {
            agg.ingest(LogEntry::new(LogLevel::Info, "msg").with_source("worker"));
        }
        let top = agg.top_k_sources(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].value, "api");
    }

    #[test]
    fn test_retention_max_entries() {
        let mut agg = LogAggregator::new()
            .with_retention(RetentionPolicy::new().with_max_entries(3));
        for i in 0..5 {
            agg.ingest(LogEntry::new(LogLevel::Info, &format!("msg {}", i)));
        }
        assert_eq!(agg.len(), 3);
    }

    #[test]
    fn test_parse_timestamp_rfc3339() {
        let ts = parse_timestamp("2025-01-15T10:30:00Z");
        assert!(ts.is_some());
    }

    #[test]
    fn test_parse_timestamp_common() {
        let ts = parse_timestamp("2025-01-15 10:30:00");
        assert!(ts.is_some());
    }

    #[test]
    fn test_extract_fields() {
        let line = "level=error msg=\"Connection failed\" host=db-1 port=5432";
        let fields = extract_fields(line);
        assert_eq!(fields.get("level"), Some(&"error".to_string()));
        assert_eq!(fields.get("msg"), Some(&"Connection failed".to_string()));
        assert_eq!(fields.get("host"), Some(&"db-1".to_string()));
    }

    #[test]
    fn test_ingest_raw() {
        let mut agg = LogAggregator::new();
        agg.ingest_raw("2025-01-15T10:30:00Z ERROR Connection timeout host=db-1");
        assert_eq!(agg.len(), 1);
        let entries = agg.query(&LogFilter::new());
        assert_eq!(entries[0].level, LogLevel::Error);
    }

    #[test]
    fn test_error_rate() {
        let mut agg = LogAggregator::new();
        for _ in 0..8 {
            agg.ingest(LogEntry::new(LogLevel::Info, "ok"));
        }
        for _ in 0..2 {
            agg.ingest(LogEntry::new(LogLevel::Error, "err"));
        }
        let rate = agg.error_rate(10);
        assert!((rate - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aggregator_clear() {
        let mut agg = LogAggregator::new();
        agg.ingest(LogEntry::new(LogLevel::Info, "msg"));
        agg.clear();
        assert!(agg.is_empty());
        assert_eq!(agg.estimated_bytes(), 0);
    }

    #[test]
    fn test_filter_with_limit() {
        let mut agg = LogAggregator::new();
        for i in 0..10 {
            agg.ingest(LogEntry::new(LogLevel::Info, &format!("msg {}", i)));
        }
        let results = agg.query(&LogFilter::new().with_limit(3));
        assert_eq!(results.len(), 3);
    }
}
