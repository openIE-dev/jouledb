//! Console logging system with levels, circular buffer, groups, tables, counters, and timers.
//!
//! Replaces browser `console.*` APIs with a pure-Rust model that supports
//! structured logging, filtering, grouping, table formatting, `console.count`,
//! and `console.time` / `console.timeEnd`.

use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};

// ── Types ──

/// Log severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        }
    }
}

/// A single log entry.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub message: String,
    pub source_file: Option<String>,
    pub source_line: Option<u32>,
    pub data: Option<Value>,
    pub group_depth: usize,
}

// ── ConsoleBuffer ──

/// Circular-buffer console logger with filtering, grouping, counts, and timers.
pub struct ConsoleBuffer {
    entries: VecDeque<LogEntry>,
    capacity: usize,
    group_depth: usize,
    group_labels: Vec<String>,
    counters: HashMap<String, u64>,
    timers: HashMap<String, DateTime<Utc>>,
}

impl ConsoleBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
            group_depth: 0,
            group_labels: Vec::new(),
            counters: HashMap::new(),
            timers: HashMap::new(),
        }
    }

    /// Log a message at a given level.
    pub fn log(&mut self, level: LogLevel, message: &str) {
        self.push_entry(level, message, None, None, None);
    }

    /// Log with source location.
    pub fn log_with_source(
        &mut self,
        level: LogLevel,
        message: &str,
        file: &str,
        line: u32,
    ) {
        self.push_entry(level, message, Some(file), Some(line), None);
    }

    /// Log with attached data.
    pub fn log_with_data(&mut self, level: LogLevel, message: &str, data: Value) {
        self.push_entry(level, message, None, None, Some(data));
    }

    fn push_entry(
        &mut self,
        level: LogLevel,
        message: &str,
        file: Option<&str>,
        line: Option<u32>,
        data: Option<Value>,
    ) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(LogEntry {
            timestamp: Utc::now(),
            level,
            message: message.to_string(),
            source_file: file.map(|s| s.to_string()),
            source_line: line,
            data,
            group_depth: self.group_depth,
        });
    }

    /// Start a new console group.
    pub fn group(&mut self, label: &str) {
        self.log(LogLevel::Info, &format!("▸ {label}"));
        self.group_labels.push(label.to_string());
        self.group_depth += 1;
    }

    /// End the current console group.
    pub fn group_end(&mut self) {
        if self.group_depth > 0 {
            self.group_depth -= 1;
            self.group_labels.pop();
        }
    }

    /// Increment a named counter and log the new count.
    pub fn count(&mut self, label: &str) -> u64 {
        let counter = self.counters.entry(label.to_string()).or_insert(0);
        *counter += 1;
        let val = *counter;
        self.log(LogLevel::Info, &format!("{label}: {val}"));
        val
    }

    /// Reset a named counter.
    pub fn count_reset(&mut self, label: &str) {
        self.counters.insert(label.to_string(), 0);
    }

    /// Start a named timer.
    pub fn time(&mut self, label: &str) {
        self.timers.insert(label.to_string(), Utc::now());
    }

    /// End a named timer and return elapsed milliseconds.
    pub fn time_end(&mut self, label: &str) -> Option<i64> {
        if let Some(start) = self.timers.remove(label) {
            let elapsed = Utc::now().signed_duration_since(start).num_milliseconds();
            self.log(LogLevel::Info, &format!("{label}: {elapsed}ms"));
            Some(elapsed)
        } else {
            None
        }
    }

    /// Format tabular data as aligned columns.
    pub fn table(&mut self, headers: &[&str], rows: &[Vec<String>]) {
        if headers.is_empty() {
            return;
        }

        // Compute column widths
        let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        for row in rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() && cell.len() > widths[i] {
                    widths[i] = cell.len();
                }
            }
        }

        // Format header
        let header_line: String = headers
            .iter()
            .enumerate()
            .map(|(i, h)| format!("{:width$}", h, width = widths[i]))
            .collect::<Vec<_>>()
            .join(" | ");

        let separator: String = widths
            .iter()
            .map(|w| "-".repeat(*w))
            .collect::<Vec<_>>()
            .join("-+-");

        let mut output = format!("{header_line}\n{separator}");

        for row in rows {
            let row_line: String = row
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    let w = widths.get(i).copied().unwrap_or(cell.len());
                    format!("{:width$}", cell, width = w)
                })
                .collect::<Vec<_>>()
                .join(" | ");
            output.push('\n');
            output.push_str(&row_line);
        }

        self.log(LogLevel::Info, &output);
    }

    /// Get all entries.
    pub fn entries(&self) -> &VecDeque<LogEntry> {
        &self.entries
    }

    /// Total entries currently buffered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Filter entries by minimum log level.
    pub fn filter_by_level(&self, min_level: LogLevel) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| e.level >= min_level).collect()
    }

    /// Filter entries by source file pattern (substring match).
    pub fn filter_by_source(&self, pattern: &str) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| {
                e.source_file
                    .as_ref()
                    .map(|f| f.contains(pattern))
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Current group depth.
    pub fn current_group_depth(&self) -> usize {
        self.group_depth
    }
}

impl Default for ConsoleBuffer {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_levels_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
    }

    #[test]
    fn test_basic_logging() {
        let mut buf = ConsoleBuffer::new(100);
        buf.log(LogLevel::Info, "hello");
        buf.log(LogLevel::Error, "fail");
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.entries()[0].message, "hello");
        assert_eq!(buf.entries()[1].level, LogLevel::Error);
    }

    #[test]
    fn test_circular_capacity() {
        let mut buf = ConsoleBuffer::new(3);
        for i in 0..5 {
            buf.log(LogLevel::Info, &format!("msg{i}"));
        }
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.entries()[0].message, "msg2");
        assert_eq!(buf.entries()[2].message, "msg4");
    }

    #[test]
    fn test_filter_by_level() {
        let mut buf = ConsoleBuffer::new(100);
        buf.log(LogLevel::Trace, "t");
        buf.log(LogLevel::Debug, "d");
        buf.log(LogLevel::Info, "i");
        buf.log(LogLevel::Warn, "w");
        buf.log(LogLevel::Error, "e");
        let warns = buf.filter_by_level(LogLevel::Warn);
        assert_eq!(warns.len(), 2);
    }

    #[test]
    fn test_filter_by_source() {
        let mut buf = ConsoleBuffer::new(100);
        buf.log_with_source(LogLevel::Info, "a", "src/main.rs", 10);
        buf.log_with_source(LogLevel::Info, "b", "src/lib.rs", 20);
        buf.log(LogLevel::Info, "c");
        let results = buf.filter_by_source("main");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].message, "a");
    }

    #[test]
    fn test_group_nesting() {
        let mut buf = ConsoleBuffer::new(100);
        assert_eq!(buf.current_group_depth(), 0);
        buf.group("outer");
        assert_eq!(buf.current_group_depth(), 1);
        buf.log(LogLevel::Info, "inside outer");
        buf.group("inner");
        assert_eq!(buf.current_group_depth(), 2);
        buf.log(LogLevel::Info, "inside inner");
        buf.group_end();
        assert_eq!(buf.current_group_depth(), 1);
        buf.group_end();
        assert_eq!(buf.current_group_depth(), 0);
        // group_end at 0 is a no-op
        buf.group_end();
        assert_eq!(buf.current_group_depth(), 0);
    }

    #[test]
    fn test_group_depth_on_entries() {
        let mut buf = ConsoleBuffer::new(100);
        buf.log(LogLevel::Info, "top");
        buf.group("g");
        buf.log(LogLevel::Info, "nested");
        assert_eq!(buf.entries()[0].group_depth, 0);
        // entries[1] is the group label, entries[2] is "nested"
        assert_eq!(buf.entries()[2].group_depth, 1);
    }

    #[test]
    fn test_count() {
        let mut buf = ConsoleBuffer::new(100);
        assert_eq!(buf.count("clicks"), 1);
        assert_eq!(buf.count("clicks"), 2);
        assert_eq!(buf.count("other"), 1);
        buf.count_reset("clicks");
        assert_eq!(buf.count("clicks"), 1);
    }

    #[test]
    fn test_timer() {
        let mut buf = ConsoleBuffer::new(100);
        buf.time("op");
        let elapsed = buf.time_end("op");
        assert!(elapsed.is_some());
        assert!(elapsed.unwrap() >= 0);
        // Double end returns None
        assert!(buf.time_end("op").is_none());
    }

    #[test]
    fn test_table_formatting() {
        let mut buf = ConsoleBuffer::new(100);
        buf.table(
            &["Name", "Age"],
            &[
                vec!["Alice".into(), "30".into()],
                vec!["Bob".into(), "25".into()],
            ],
        );
        assert_eq!(buf.len(), 1);
        let msg = &buf.entries()[0].message;
        assert!(msg.contains("Name"));
        assert!(msg.contains("Alice"));
        assert!(msg.contains("---"));
    }

    #[test]
    fn test_log_with_data() {
        let mut buf = ConsoleBuffer::new(100);
        buf.log_with_data(
            LogLevel::Debug,
            "payload",
            serde_json::json!({"key": "val"}),
        );
        let entry = &buf.entries()[0];
        assert!(entry.data.is_some());
        assert_eq!(entry.data.as_ref().unwrap()["key"], "val");
    }

    #[test]
    fn test_clear() {
        let mut buf = ConsoleBuffer::new(100);
        buf.log(LogLevel::Info, "a");
        buf.log(LogLevel::Info, "b");
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_level_as_str() {
        assert_eq!(LogLevel::Error.as_str(), "ERROR");
        assert_eq!(LogLevel::Trace.as_str(), "TRACE");
    }
}
