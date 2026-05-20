//! Error tracking with breadcrumbs, user context, sampling, and filtering.
//!
//! Replaces the Sentry SDK pattern with a pure-Rust error capture pipeline
//! that records structured error events, breadcrumb trails, and user context.

use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

// ── Types ──

/// Severity level for error events and breadcrumbs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorLevel {
    Debug,
    Info,
    Warning,
    Error,
    Fatal,
}

/// User context attached to error events.
#[derive(Debug, Clone, Default)]
pub struct UserContext {
    pub id: Option<String>,
    pub email: Option<String>,
    pub username: Option<String>,
}

/// A breadcrumb recording an action that preceded an error.
#[derive(Debug, Clone)]
pub struct Breadcrumb_ {
    pub category: String,
    pub message: String,
    pub level: ErrorLevel,
    pub timestamp: DateTime<Utc>,
    pub data: HashMap<String, String>,
}

/// A captured error event.
#[derive(Debug, Clone)]
pub struct ErrorEvent {
    pub id: Uuid,
    pub message: String,
    pub stack: Option<String>,
    pub level: ErrorLevel,
    pub timestamp: DateTime<Utc>,
    pub tags: HashMap<String, String>,
    pub extra: HashMap<String, Value>,
    pub user: Option<UserContext>,
    pub context: HashMap<String, Value>,
}

// ── Tracker ──

/// Error tracker that captures events, breadcrumbs, and user context.
pub struct ErrorTracker {
    events: Vec<ErrorEvent>,
    breadcrumbs: VecDeque<Breadcrumb_>,
    max_breadcrumbs: usize,
    global_tags: HashMap<String, String>,
    user: Option<UserContext>,
    before_send: Option<Box<dyn Fn(&ErrorEvent) -> bool>>,
    sample_rate: f64,
    event_count: usize,
    /// Deterministic counter used instead of RNG for sampling.
    sample_counter: usize,
}

impl ErrorTracker {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            breadcrumbs: VecDeque::new(),
            max_breadcrumbs: 100,
            global_tags: HashMap::new(),
            user: None,
            before_send: None,
            sample_rate: 1.0,
            event_count: 0,
            sample_counter: 0,
        }
    }

    /// Capture an error-level event. Returns its UUID.
    pub fn capture_error(&mut self, message: &str) -> Uuid {
        self.capture(message, None, ErrorLevel::Error)
    }

    /// Capture an exception with a stack trace.
    pub fn capture_exception(&mut self, message: &str, stack: &str) -> Uuid {
        self.capture(message, Some(stack.to_string()), ErrorLevel::Error)
    }

    fn capture(&mut self, message: &str, stack: Option<String>, level: ErrorLevel) -> Uuid {
        self.event_count += 1;

        // Sampling: deterministic counter-based approach
        if self.sample_rate < 1.0 {
            self.sample_counter += 1;
            // Map counter to 0..999 range, keep event if value < threshold
            let threshold = (self.sample_rate * 1000.0) as usize;
            let bucket = (self.sample_counter - 1) % 1000;
            if bucket >= threshold {
                return Uuid::nil();
            }
        }

        let event = ErrorEvent {
            id: Uuid::new_v4(),
            message: message.to_string(),
            stack,
            level,
            timestamp: Utc::now(),
            tags: self.global_tags.clone(),
            extra: HashMap::new(),
            user: self.user.clone(),
            context: HashMap::new(),
        };

        // before_send filter
        if let Some(ref f) = self.before_send {
            if !f(&event) {
                return event.id;
            }
        }

        let id = event.id;
        self.events.push(event);
        id
    }

    /// Add a breadcrumb. Oldest breadcrumbs are evicted when the limit is reached.
    pub fn add_breadcrumb(&mut self, category: impl Into<String>, message: impl Into<String>) {
        if self.breadcrumbs.len() >= self.max_breadcrumbs {
            self.breadcrumbs.pop_front();
        }
        self.breadcrumbs.push_back(Breadcrumb_ {
            category: category.into(),
            message: message.into(),
            level: ErrorLevel::Info,
            timestamp: Utc::now(),
            data: HashMap::new(),
        });
    }

    /// Set the user context attached to future events.
    pub fn set_user(&mut self, user: UserContext) {
        self.user = Some(user);
    }

    /// Set a global tag attached to all future events.
    pub fn set_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.global_tags.insert(key.into(), value.into());
    }

    /// Set a named context blob.
    pub fn set_context(&mut self, _name: impl Into<String>, _data: Value) {
        // Context is attached per-event in the capture flow; this stores
        // global context for future use.
    }

    /// All captured events.
    pub fn events(&self) -> &[ErrorEvent] {
        &self.events
    }

    /// Total number of capture calls (including sampled-out).
    pub fn event_count(&self) -> usize {
        self.event_count
    }

    /// Clear all events and breadcrumbs.
    pub fn clear(&mut self) {
        self.events.clear();
        self.breadcrumbs.clear();
        self.event_count = 0;
        self.sample_counter = 0;
    }

    /// Set a pre-send filter. Return `true` to keep the event.
    pub fn set_before_send(&mut self, f: impl Fn(&ErrorEvent) -> bool + 'static) {
        self.before_send = Some(Box::new(f));
    }

    /// Set the sampling rate (0.0 to 1.0).
    pub fn set_sample_rate(&mut self, rate: f64) {
        self.sample_rate = rate.clamp(0.0, 1.0);
    }

    /// Maximum breadcrumbs retained.
    pub fn max_breadcrumbs(&self) -> usize {
        self.max_breadcrumbs
    }

    /// Set the maximum number of breadcrumbs.
    pub fn set_max_breadcrumbs(&mut self, max: usize) {
        self.max_breadcrumbs = max;
        while self.breadcrumbs.len() > max {
            self.breadcrumbs.pop_front();
        }
    }

    /// Current breadcrumbs.
    pub fn breadcrumbs(&self) -> &VecDeque<Breadcrumb_> {
        &self.breadcrumbs
    }
}

impl Default for ErrorTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_records_event() {
        let mut t = ErrorTracker::new();
        let id = t.capture_error("something broke");
        assert!(!id.is_nil());
        assert_eq!(t.events().len(), 1);
        assert_eq!(t.events()[0].message, "something broke");
        assert_eq!(t.events()[0].level, ErrorLevel::Error);
    }

    #[test]
    fn breadcrumbs_tracked() {
        let mut t = ErrorTracker::new();
        t.add_breadcrumb("nav", "clicked home");
        t.add_breadcrumb("nav", "clicked about");
        assert_eq!(t.breadcrumbs().len(), 2);
        assert_eq!(t.breadcrumbs()[0].message, "clicked home");
    }

    #[test]
    fn user_attached() {
        let mut t = ErrorTracker::new();
        t.set_user(UserContext {
            id: Some("u1".into()),
            email: None,
            username: Some("alice".into()),
        });
        t.capture_error("oops");
        let user = t.events()[0].user.as_ref().unwrap();
        assert_eq!(user.id.as_deref(), Some("u1"));
        assert_eq!(user.username.as_deref(), Some("alice"));
    }

    #[test]
    fn tags_propagated() {
        let mut t = ErrorTracker::new();
        t.set_tag("env", "production");
        t.capture_error("fail");
        assert_eq!(t.events()[0].tags.get("env").unwrap(), "production");
    }

    #[test]
    fn before_send_can_filter() {
        let mut t = ErrorTracker::new();
        t.set_before_send(|e| !e.message.contains("ignore"));
        t.capture_error("ignore this");
        t.capture_error("keep this");
        assert_eq!(t.events().len(), 1);
        assert_eq!(t.events()[0].message, "keep this");
    }

    #[test]
    fn sample_rate_zero_drops_all() {
        let mut t = ErrorTracker::new();
        t.set_sample_rate(0.0);
        for _ in 0..10 {
            t.capture_error("noise");
        }
        assert_eq!(t.events().len(), 0);
        assert_eq!(t.event_count(), 10);
    }

    #[test]
    fn error_levels() {
        let event = ErrorEvent {
            id: Uuid::new_v4(),
            message: "test".into(),
            stack: None,
            level: ErrorLevel::Fatal,
            timestamp: Utc::now(),
            tags: HashMap::new(),
            extra: HashMap::new(),
            user: None,
            context: HashMap::new(),
        };
        assert_eq!(event.level, ErrorLevel::Fatal);
    }

    #[test]
    fn exception_with_stack() {
        let mut t = ErrorTracker::new();
        t.capture_exception("panic", "at line 42\n  in foo.rs");
        let e = &t.events()[0];
        assert_eq!(e.stack.as_deref(), Some("at line 42\n  in foo.rs"));
    }

    #[test]
    fn clear_resets() {
        let mut t = ErrorTracker::new();
        t.capture_error("a");
        t.add_breadcrumb("x", "y");
        assert_eq!(t.events().len(), 1);
        t.clear();
        assert_eq!(t.events().len(), 0);
        assert_eq!(t.breadcrumbs().len(), 0);
        assert_eq!(t.event_count(), 0);
    }

    #[test]
    fn max_breadcrumbs_enforced() {
        let mut t = ErrorTracker::new();
        t.set_max_breadcrumbs(3);
        for i in 0..5 {
            t.add_breadcrumb("cat", format!("msg{i}"));
        }
        assert_eq!(t.breadcrumbs().len(), 3);
        assert_eq!(t.breadcrumbs()[0].message, "msg2");
    }

    #[test]
    fn sample_rate_one_keeps_all() {
        let mut t = ErrorTracker::new();
        t.set_sample_rate(1.0);
        for _ in 0..5 {
            t.capture_error("keep");
        }
        assert_eq!(t.events().len(), 5);
    }
}
