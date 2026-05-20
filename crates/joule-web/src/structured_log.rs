//! Structured logging: JSON log format, context/span propagation, log levels,
//! field builders, correlation IDs, sampling, and ELK/Datadog compatible output.
//!
//! Pure Rust — no I/O, no filesystem. Log entries are built in memory
//! and serialized to JSON strings. A `LogSink` collects entries for
//! assertion or downstream processing.

use std::collections::HashMap;
use std::fmt;

// ── Log level ─────────────────────────────────────────────────────

/// Severity level for log entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Fatal = 5,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Trace => write!(f, "TRACE"),
            Self::Debug => write!(f, "DEBUG"),
            Self::Info => write!(f, "INFO"),
            Self::Warn => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Fatal => write!(f, "FATAL"),
        }
    }
}

impl LogLevel {
    /// Parse from a string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "TRACE" => Some(Self::Trace),
            "DEBUG" => Some(Self::Debug),
            "INFO" => Some(Self::Info),
            "WARN" | "WARNING" => Some(Self::Warn),
            "ERROR" | "ERR" => Some(Self::Error),
            "FATAL" | "CRITICAL" => Some(Self::Fatal),
            _ => None,
        }
    }
}

// ── Log field ─────────────────────────────────────────────────────

/// A typed field attached to a log entry.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl fmt::Display for FieldValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Str(s) => write!(f, "{s}"),
            Self::Int(i) => write!(f, "{i}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Bool(b) => write!(f, "{b}"),
        }
    }
}

impl FieldValue {
    /// Render as a JSON value fragment.
    pub fn to_json_value(&self) -> String {
        match self {
            Self::Str(s) => format!(r#""{}""#, s.replace('\\', r"\\").replace('"', r#"\""#)),
            Self::Int(i) => i.to_string(),
            Self::Float(v) => format!("{v}"),
            Self::Bool(b) => b.to_string(),
        }
    }
}

// ── Log context ───────────────────────────────────────────────────

/// Contextual fields propagated to all log entries (span-style).
#[derive(Debug, Clone, Default)]
pub struct LogContext {
    /// Correlation / request ID for tracing across services.
    pub correlation_id: Option<String>,
    /// Trace ID from distributed tracing.
    pub trace_id: Option<String>,
    /// Span ID from distributed tracing.
    pub span_id: Option<String>,
    /// Additional context fields inherited by all log entries.
    pub fields: HashMap<String, FieldValue>,
}

impl LogContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the correlation ID.
    pub fn with_correlation_id(mut self, id: &str) -> Self {
        self.correlation_id = Some(id.to_string());
        self
    }

    /// Set trace/span IDs.
    pub fn with_trace(mut self, trace_id: &str, span_id: &str) -> Self {
        self.trace_id = Some(trace_id.to_string());
        self.span_id = Some(span_id.to_string());
        self
    }

    /// Add a context field.
    pub fn with_field(mut self, key: &str, value: FieldValue) -> Self {
        self.fields.insert(key.to_string(), value);
        self
    }
}

// ── Log entry ─────────────────────────────────────────────────────

/// A single structured log entry.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Timestamp as ISO-8601 string (caller-supplied for purity).
    pub timestamp: String,
    /// Severity level.
    pub level: LogLevel,
    /// Human-readable message.
    pub message: String,
    /// Module or logger name.
    pub module: Option<String>,
    /// Structured fields.
    pub fields: HashMap<String, FieldValue>,
    /// Correlation ID (copied from context or set directly).
    pub correlation_id: Option<String>,
    /// Trace ID.
    pub trace_id: Option<String>,
    /// Span ID.
    pub span_id: Option<String>,
}

impl LogEntry {
    /// Render as a JSON string (ELK/Datadog compatible).
    pub fn to_json(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        parts.push(format!(r#""timestamp":"{}""#, self.timestamp));
        parts.push(format!(r#""level":"{}""#, self.level));
        parts.push(format!(
            r#""message":"{}""#,
            self.message.replace('\\', r"\\").replace('"', r#"\""#)
        ));

        if let Some(ref m) = self.module {
            parts.push(format!(r#""module":"{}""#, m));
        }
        if let Some(ref cid) = self.correlation_id {
            parts.push(format!(r#""correlation_id":"{}""#, cid));
        }
        if let Some(ref tid) = self.trace_id {
            parts.push(format!(r#""trace_id":"{}""#, tid));
        }
        if let Some(ref sid) = self.span_id {
            parts.push(format!(r#""span_id":"{}""#, sid));
        }

        // Sort field keys for deterministic output.
        let mut field_keys: Vec<&String> = self.fields.keys().collect();
        field_keys.sort();
        for key in field_keys {
            let val = &self.fields[key];
            parts.push(format!(r#""{}":{}"#, key, val.to_json_value()));
        }

        format!("{{{}}}", parts.join(","))
    }
}

// ── Log entry builder ─────────────────────────────────────────────

/// Builder for constructing log entries with a fluent API.
pub struct LogEntryBuilder {
    timestamp: String,
    level: LogLevel,
    message: String,
    module: Option<String>,
    fields: HashMap<String, FieldValue>,
    correlation_id: Option<String>,
    trace_id: Option<String>,
    span_id: Option<String>,
}

impl LogEntryBuilder {
    /// Start building an entry.
    pub fn new(level: LogLevel, message: &str) -> Self {
        Self {
            timestamp: String::new(),
            level,
            message: message.to_string(),
            module: None,
            fields: HashMap::new(),
            correlation_id: None,
            trace_id: None,
            span_id: None,
        }
    }

    /// Set the timestamp.
    pub fn timestamp(mut self, ts: &str) -> Self {
        self.timestamp = ts.to_string();
        self
    }

    /// Set the module name.
    pub fn module(mut self, m: &str) -> Self {
        self.module = Some(m.to_string());
        self
    }

    /// Add a string field.
    pub fn field_str(mut self, key: &str, value: &str) -> Self {
        self.fields
            .insert(key.to_string(), FieldValue::Str(value.to_string()));
        self
    }

    /// Add an integer field.
    pub fn field_int(mut self, key: &str, value: i64) -> Self {
        self.fields
            .insert(key.to_string(), FieldValue::Int(value));
        self
    }

    /// Add a float field.
    pub fn field_float(mut self, key: &str, value: f64) -> Self {
        self.fields
            .insert(key.to_string(), FieldValue::Float(value));
        self
    }

    /// Add a boolean field.
    pub fn field_bool(mut self, key: &str, value: bool) -> Self {
        self.fields
            .insert(key.to_string(), FieldValue::Bool(value));
        self
    }

    /// Apply context fields (correlation, trace, span, and custom fields).
    pub fn with_context(mut self, ctx: &LogContext) -> Self {
        if let Some(ref cid) = ctx.correlation_id {
            self.correlation_id = Some(cid.clone());
        }
        if let Some(ref tid) = ctx.trace_id {
            self.trace_id = Some(tid.clone());
        }
        if let Some(ref sid) = ctx.span_id {
            self.span_id = Some(sid.clone());
        }
        for (k, v) in &ctx.fields {
            self.fields.entry(k.clone()).or_insert_with(|| v.clone());
        }
        self
    }

    /// Build the log entry.
    pub fn build(self) -> LogEntry {
        LogEntry {
            timestamp: self.timestamp,
            level: self.level,
            message: self.message,
            module: self.module,
            fields: self.fields,
            correlation_id: self.correlation_id,
            trace_id: self.trace_id,
            span_id: self.span_id,
        }
    }
}

// ── Sampling ──────────────────────────────────────────────────────

/// Sampling strategy for reducing log volume.
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Log everything.
    Always,
    /// Log 1 in N entries.
    RateBased(u64),
    /// Log entries at or above a given level; sample below.
    LevelBased { always_above: LogLevel, sample_rate: u64 },
}

impl SamplingStrategy {
    /// Decide whether to emit a log entry given a counter and level.
    pub fn should_emit(&self, counter: u64, level: LogLevel) -> bool {
        match self {
            Self::Always => true,
            Self::RateBased(n) => {
                if *n == 0 {
                    return false;
                }
                counter % n == 0
            }
            Self::LevelBased {
                always_above,
                sample_rate,
            } => {
                if level >= *always_above {
                    true
                } else if *sample_rate == 0 {
                    false
                } else {
                    counter % sample_rate == 0
                }
            }
        }
    }
}

// ── Log sink ──────────────────────────────────────────────────────

/// In-memory log collector for structured log entries.
#[derive(Debug, Clone)]
pub struct LogSink {
    entries: Vec<LogEntry>,
    min_level: LogLevel,
    module_filters: Vec<(String, LogLevel)>,
    sampling: SamplingStrategy,
    counter: u64,
    context: LogContext,
}

impl LogSink {
    /// Create a new sink accepting entries at or above `min_level`.
    pub fn new(min_level: LogLevel) -> Self {
        Self {
            entries: Vec::new(),
            min_level,
            module_filters: Vec::new(),
            sampling: SamplingStrategy::Always,
            counter: 0,
            context: LogContext::new(),
        }
    }

    /// Set the global context for all entries.
    pub fn set_context(&mut self, ctx: LogContext) {
        self.context = ctx;
    }

    /// Set sampling strategy.
    pub fn set_sampling(&mut self, strategy: SamplingStrategy) {
        self.sampling = strategy;
    }

    /// Add a module-level filter (entries from this module need at least this level).
    pub fn add_module_filter(&mut self, module_prefix: &str, level: LogLevel) {
        self.module_filters
            .push((module_prefix.to_string(), level));
    }

    /// Emit a log entry (applies filters and sampling).
    pub fn emit(&mut self, mut entry: LogEntry) {
        // Level gate.
        if entry.level < self.min_level {
            return;
        }

        // Module filter.
        if let Some(ref module) = entry.module {
            for (prefix, level) in &self.module_filters {
                if module.starts_with(prefix) && entry.level < *level {
                    return;
                }
            }
        }

        // Sampling.
        self.counter = self.counter.wrapping_add(1);
        if !self.sampling.should_emit(self.counter, entry.level) {
            return;
        }

        // Apply context.
        if entry.correlation_id.is_none() {
            entry.correlation_id = self.context.correlation_id.clone();
        }
        if entry.trace_id.is_none() {
            entry.trace_id = self.context.trace_id.clone();
        }
        if entry.span_id.is_none() {
            entry.span_id = self.context.span_id.clone();
        }
        for (k, v) in &self.context.fields {
            entry.fields.entry(k.clone()).or_insert_with(|| v.clone());
        }

        self.entries.push(entry);
    }

    /// Emit using the builder pattern (convenience).
    pub fn log(&mut self, level: LogLevel, message: &str, timestamp: &str) {
        let entry = LogEntryBuilder::new(level, message)
            .timestamp(timestamp)
            .build();
        self.emit(entry);
    }

    /// Get all collected entries.
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Get entries as JSON lines (one JSON object per line).
    pub fn to_json_lines(&self) -> String {
        self.entries
            .iter()
            .map(|e| e.to_json())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Count entries by level.
    pub fn count_by_level(&self) -> HashMap<LogLevel, usize> {
        let mut counts = HashMap::new();
        for entry in &self.entries {
            *counts.entry(entry.level).or_insert(0) += 1;
        }
        counts
    }

    /// Clear all collected entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Total number of collected entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for LogSink {
    fn default() -> Self {
        Self::new(LogLevel::Trace)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Fatal);
    }

    #[test]
    fn level_display() {
        assert_eq!(LogLevel::Info.to_string(), "INFO");
        assert_eq!(LogLevel::Error.to_string(), "ERROR");
    }

    #[test]
    fn level_from_str_loose() {
        assert_eq!(LogLevel::from_str_loose("info"), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_str_loose("WARNING"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str_loose("ERR"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_str_loose("CRITICAL"), Some(LogLevel::Fatal));
        assert_eq!(LogLevel::from_str_loose("nope"), None);
    }

    #[test]
    fn field_value_display() {
        assert_eq!(FieldValue::Str("hello".into()).to_string(), "hello");
        assert_eq!(FieldValue::Int(42).to_string(), "42");
        assert_eq!(FieldValue::Bool(true).to_string(), "true");
    }

    #[test]
    fn field_value_json() {
        assert_eq!(FieldValue::Str("hi".into()).to_json_value(), r#""hi""#);
        assert_eq!(FieldValue::Int(7).to_json_value(), "7");
        assert_eq!(FieldValue::Bool(false).to_json_value(), "false");
    }

    #[test]
    fn field_value_json_escaping() {
        let v = FieldValue::Str(r#"say "hello""#.into());
        let json = v.to_json_value();
        assert!(json.contains(r#"\""#));
    }

    #[test]
    fn log_entry_to_json_basic() {
        let entry = LogEntryBuilder::new(LogLevel::Info, "request handled")
            .timestamp("2026-03-09T12:00:00Z")
            .module("http")
            .field_int("status", 200)
            .field_str("method", "GET")
            .build();
        let json = entry.to_json();
        assert!(json.contains(r#""level":"INFO""#));
        assert!(json.contains(r#""message":"request handled""#));
        assert!(json.contains(r#""module":"http""#));
        assert!(json.contains(r#""status":200"#));
        assert!(json.contains(r#""method":"GET""#));
    }

    #[test]
    fn log_entry_with_context() {
        let ctx = LogContext::new()
            .with_correlation_id("req-123")
            .with_trace("trace-abc", "span-def")
            .with_field("service", FieldValue::Str("api".into()));

        let entry = LogEntryBuilder::new(LogLevel::Warn, "slow query")
            .timestamp("2026-03-09T12:00:00Z")
            .with_context(&ctx)
            .build();

        let json = entry.to_json();
        assert!(json.contains(r#""correlation_id":"req-123""#));
        assert!(json.contains(r#""trace_id":"trace-abc""#));
        assert!(json.contains(r#""span_id":"span-def""#));
        assert!(json.contains(r#""service":"api""#));
    }

    #[test]
    fn sink_filters_by_level() {
        let mut sink = LogSink::new(LogLevel::Warn);
        sink.log(LogLevel::Debug, "should be filtered", "t0");
        sink.log(LogLevel::Info, "also filtered", "t1");
        sink.log(LogLevel::Warn, "should pass", "t2");
        sink.log(LogLevel::Error, "also pass", "t3");
        assert_eq!(sink.len(), 2);
    }

    #[test]
    fn sink_module_filter() {
        let mut sink = LogSink::new(LogLevel::Trace);
        sink.add_module_filter("noisy", LogLevel::Error);

        let noisy_debug = LogEntryBuilder::new(LogLevel::Debug, "noisy debug")
            .timestamp("t0")
            .module("noisy::sub")
            .build();
        sink.emit(noisy_debug);

        let noisy_error = LogEntryBuilder::new(LogLevel::Error, "noisy error")
            .timestamp("t1")
            .module("noisy::sub")
            .build();
        sink.emit(noisy_error);

        let other_debug = LogEntryBuilder::new(LogLevel::Debug, "other debug")
            .timestamp("t2")
            .module("other")
            .build();
        sink.emit(other_debug);

        assert_eq!(sink.len(), 2);
        assert_eq!(sink.entries()[0].message, "noisy error");
        assert_eq!(sink.entries()[1].message, "other debug");
    }

    #[test]
    fn sink_context_propagation() {
        let mut sink = LogSink::new(LogLevel::Trace);
        sink.set_context(
            LogContext::new()
                .with_correlation_id("global-req-1")
                .with_field("env", FieldValue::Str("prod".into())),
        );

        sink.log(LogLevel::Info, "test", "t0");
        let entry = &sink.entries()[0];
        assert_eq!(entry.correlation_id.as_deref(), Some("global-req-1"));
        assert_eq!(
            entry.fields.get("env"),
            Some(&FieldValue::Str("prod".into()))
        );
    }

    #[test]
    fn sink_entry_context_overrides_global() {
        let mut sink = LogSink::new(LogLevel::Trace);
        sink.set_context(LogContext::new().with_correlation_id("global"));

        let entry = LogEntryBuilder::new(LogLevel::Info, "custom")
            .timestamp("t0")
            .build();
        // entry has no correlation_id → will inherit global
        let mut e2 = entry;
        e2.correlation_id = Some("local".into());
        sink.emit(e2);
        assert_eq!(
            sink.entries()[0].correlation_id.as_deref(),
            Some("local")
        );
    }

    #[test]
    fn sampling_always() {
        let s = SamplingStrategy::Always;
        for i in 0..10 {
            assert!(s.should_emit(i, LogLevel::Info));
        }
    }

    #[test]
    fn sampling_rate_based() {
        let s = SamplingStrategy::RateBased(3);
        let results: Vec<bool> = (0..9).map(|i| s.should_emit(i, LogLevel::Info)).collect();
        // 0,3,6 → true; 1,2,4,5,7,8 → false
        assert_eq!(
            results,
            vec![true, false, false, true, false, false, true, false, false]
        );
    }

    #[test]
    fn sampling_rate_zero() {
        let s = SamplingStrategy::RateBased(0);
        assert!(!s.should_emit(0, LogLevel::Info));
    }

    #[test]
    fn sampling_level_based() {
        let s = SamplingStrategy::LevelBased {
            always_above: LogLevel::Error,
            sample_rate: 10,
        };
        // Error always emitted.
        assert!(s.should_emit(7, LogLevel::Error));
        // Info only when counter % 10 == 0.
        assert!(s.should_emit(0, LogLevel::Info));
        assert!(!s.should_emit(1, LogLevel::Info));
    }

    #[test]
    fn sampling_level_based_zero_rate() {
        let s = SamplingStrategy::LevelBased {
            always_above: LogLevel::Error,
            sample_rate: 0,
        };
        assert!(!s.should_emit(0, LogLevel::Info));
        assert!(s.should_emit(0, LogLevel::Error));
    }

    #[test]
    fn sink_sampling() {
        let mut sink = LogSink::new(LogLevel::Trace);
        sink.set_sampling(SamplingStrategy::RateBased(2));
        // Counter starts at 0, increments before check.
        // emit 1: counter=1, 1%2=1 → skip
        // emit 2: counter=2, 2%2=0 → emit
        // emit 3: counter=3, 3%2=1 → skip
        // emit 4: counter=4, 4%2=0 → emit
        for _ in 0..4 {
            sink.log(LogLevel::Info, "msg", "t");
        }
        assert_eq!(sink.len(), 2);
    }

    #[test]
    fn count_by_level() {
        let mut sink = LogSink::new(LogLevel::Trace);
        sink.log(LogLevel::Info, "a", "t0");
        sink.log(LogLevel::Info, "b", "t1");
        sink.log(LogLevel::Error, "c", "t2");
        let counts = sink.count_by_level();
        assert_eq!(counts.get(&LogLevel::Info).copied(), Some(2));
        assert_eq!(counts.get(&LogLevel::Error).copied(), Some(1));
        assert_eq!(counts.get(&LogLevel::Debug).copied(), None);
    }

    #[test]
    fn clear_entries() {
        let mut sink = LogSink::new(LogLevel::Trace);
        sink.log(LogLevel::Info, "msg", "t");
        assert!(!sink.is_empty());
        sink.clear();
        assert!(sink.is_empty());
        assert_eq!(sink.len(), 0);
    }

    #[test]
    fn json_lines_format() {
        let mut sink = LogSink::new(LogLevel::Trace);
        sink.log(LogLevel::Info, "first", "t0");
        sink.log(LogLevel::Warn, "second", "t1");
        let lines = sink.to_json_lines();
        let parts: Vec<&str> = lines.split('\n').collect();
        assert_eq!(parts.len(), 2);
        assert!(parts[0].contains("first"));
        assert!(parts[1].contains("second"));
    }

    #[test]
    fn builder_all_field_types() {
        let entry = LogEntryBuilder::new(LogLevel::Debug, "test")
            .timestamp("t")
            .field_str("s", "val")
            .field_int("i", 42)
            .field_float("f", 3.14)
            .field_bool("b", true)
            .build();
        assert_eq!(
            entry.fields.get("s"),
            Some(&FieldValue::Str("val".into()))
        );
        assert_eq!(entry.fields.get("i"), Some(&FieldValue::Int(42)));
        assert_eq!(entry.fields.get("f"), Some(&FieldValue::Float(3.14)));
        assert_eq!(entry.fields.get("b"), Some(&FieldValue::Bool(true)));
    }

    #[test]
    fn log_context_default() {
        let ctx = LogContext::new();
        assert!(ctx.correlation_id.is_none());
        assert!(ctx.trace_id.is_none());
        assert!(ctx.span_id.is_none());
        assert!(ctx.fields.is_empty());
    }

    #[test]
    fn json_message_escaping() {
        let entry = LogEntryBuilder::new(LogLevel::Info, r#"said "hello""#)
            .timestamp("t")
            .build();
        let json = entry.to_json();
        assert!(json.contains(r#"said \""#));
    }

    #[test]
    fn default_sink() {
        let sink = LogSink::default();
        assert_eq!(sink.min_level, LogLevel::Trace);
        assert!(sink.is_empty());
    }

    #[test]
    fn context_field_not_overridden_by_entry() {
        let mut sink = LogSink::new(LogLevel::Trace);
        sink.set_context(
            LogContext::new().with_field("env", FieldValue::Str("prod".into())),
        );
        let entry = LogEntryBuilder::new(LogLevel::Info, "msg")
            .timestamp("t")
            .field_str("env", "dev")
            .build();
        sink.emit(entry);
        // Entry's own field takes precedence.
        assert_eq!(
            sink.entries()[0].fields.get("env"),
            Some(&FieldValue::Str("dev".into()))
        );
    }

    #[test]
    fn float_field_json() {
        let fv = FieldValue::Float(2.718);
        let j = fv.to_json_value();
        assert!(j.starts_with("2.718"));
    }
}
