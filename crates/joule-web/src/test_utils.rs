//! Testing utilities for joule-web.
//!
//! Replaces JavaScript testing-library patterns with pure-Rust helpers:
//!
//! - [`MockTimer`] — deterministic, manually-advanced clock
//! - [`EventRecorder`] — captures timestamped events for assertions
//! - [`MockRandom`] — predetermined random-value sequence
//! - [`Snapshot`] — debug-based snapshot testing
//! - [`TestContext`] — convenience bundle of timer + random + recorder
//! - [`assert_within`] — floating-point epsilon comparison
//! - [`wait_for`] — simulated polling (no real sleep)

use std::collections::VecDeque;
use std::fmt;

// ── MockTimer ───────────────────────────────────────────────────────

/// Deterministic clock that only advances when told to.
pub struct MockTimer {
    current_ms: u64,
}

impl MockTimer {
    pub fn new() -> Self {
        Self { current_ms: 0 }
    }

    /// Advance the clock by `ms` milliseconds.
    pub fn advance(&mut self, ms: u64) {
        self.current_ms = self.current_ms.saturating_add(ms);
    }

    /// Current time in milliseconds.
    pub fn now_ms(&self) -> u64 {
        self.current_ms
    }

    /// Jump to an absolute time.
    pub fn set(&mut self, ms: u64) {
        self.current_ms = ms;
    }
}

impl Default for MockTimer {
    fn default() -> Self {
        Self::new()
    }
}

// ── EventRecorder ───────────────────────────────────────────────────

/// Records timestamped events for later assertions.
pub struct EventRecorder<T> {
    events: Vec<(T, u64)>,
}

impl<T> EventRecorder<T> {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn record(&mut self, event: T, time_ms: u64) {
        self.events.push((event, time_ms));
    }

    pub fn events(&self) -> &[(T, u64)] {
        &self.events
    }

    pub fn count(&self) -> usize {
        self.events.len()
    }

    pub fn last(&self) -> Option<&(T, u64)> {
        self.events.last()
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    pub fn contains(&self, pred: impl Fn(&T) -> bool) -> bool {
        self.events.iter().any(|(e, _)| pred(e))
    }
}

impl<T> Default for EventRecorder<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── MockRandom ──────────────────────────────────────────────────────

/// Returns predetermined f64 values in order, wrapping around when
/// the sequence is exhausted.
pub struct MockRandom {
    values: VecDeque<f64>,
    original: Vec<f64>,
}

impl MockRandom {
    pub fn new(values: Vec<f64>) -> Self {
        let original = values.clone();
        Self {
            values: VecDeque::from(values),
            original,
        }
    }

    /// Return the next predetermined value.  Wraps to the beginning
    /// when the sequence is exhausted.
    pub fn next(&mut self) -> f64 {
        if self.values.is_empty() {
            // Refill from original
            self.values = VecDeque::from(self.original.clone());
        }
        self.values.pop_front().unwrap_or(0.0)
    }
}

// ── Snapshot ────────────────────────────────────────────────────────

/// Debug-based snapshot for quick equality checks.
pub struct Snapshot {
    data: String,
}

impl Snapshot {
    /// Capture a snapshot of any `Debug` value.
    pub fn capture(value: &impl fmt::Debug) -> Self {
        Self {
            data: format!("{:?}", value),
        }
    }

    /// Compare two snapshots for equality.
    pub fn assert_eq(&self, other: &Snapshot) -> bool {
        self.data == other.data
    }

    /// Return the snapshot string.
    pub fn to_string(&self) -> &str {
        &self.data
    }
}

// ── Floating-point comparison ───────────────────────────────────────

/// Returns `true` if `|actual - expected| <= epsilon`.
pub fn assert_within(actual: f64, expected: f64, epsilon: f64) -> bool {
    (actual - expected).abs() <= epsilon
}

// ── TestContext ──────────────────────────────────────────────────────

/// Convenience bundle of frequently used test doubles.
pub struct TestContext {
    pub timer: MockTimer,
    pub random: MockRandom,
    pub events: EventRecorder<String>,
}

impl TestContext {
    pub fn new() -> Self {
        Self {
            timer: MockTimer::new(),
            random: MockRandom::new(vec![0.0, 0.25, 0.5, 0.75, 1.0]),
            events: EventRecorder::new(),
        }
    }
}

impl Default for TestContext {
    fn default() -> Self {
        Self::new()
    }
}

// ── Simulated polling ───────────────────────────────────────────────

/// Poll `condition` every `check_interval_ms` (simulated) up to
/// `max_ms`.  Returns `true` if the condition becomes true within
/// the deadline.
///
/// **Note:** This does NOT sleep; it simply calls `condition()`
/// repeatedly and compares simulated elapsed time.
pub fn wait_for(
    mut condition: impl FnMut() -> bool,
    max_ms: u64,
    check_interval_ms: u64,
) -> bool {
    let interval = check_interval_ms.max(1);
    let mut elapsed = 0u64;
    while elapsed <= max_ms {
        if condition() {
            return true;
        }
        elapsed += interval;
    }
    false
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_timer_advances() {
        let mut t = MockTimer::new();
        assert_eq!(t.now_ms(), 0);
        t.advance(100);
        assert_eq!(t.now_ms(), 100);
        t.advance(50);
        assert_eq!(t.now_ms(), 150);
    }

    #[test]
    fn mock_timer_set() {
        let mut t = MockTimer::new();
        t.set(5000);
        assert_eq!(t.now_ms(), 5000);
    }

    #[test]
    fn event_recorder_tracks() {
        let mut rec = EventRecorder::<String>::new();
        rec.record("click".into(), 100);
        rec.record("scroll".into(), 200);
        assert_eq!(rec.count(), 2);
        assert_eq!(rec.last().unwrap().0, "scroll");
        assert!(rec.contains(|e| e == "click"));
        assert!(!rec.contains(|e| e == "keydown"));
    }

    #[test]
    fn event_recorder_clear() {
        let mut rec = EventRecorder::<i32>::new();
        rec.record(1, 0);
        rec.record(2, 10);
        rec.clear();
        assert_eq!(rec.count(), 0);
    }

    #[test]
    fn mock_random_returns_sequence() {
        let mut r = MockRandom::new(vec![0.1, 0.2, 0.3]);
        assert!((r.next() - 0.1).abs() < f64::EPSILON);
        assert!((r.next() - 0.2).abs() < f64::EPSILON);
        assert!((r.next() - 0.3).abs() < f64::EPSILON);
        // Wraps around
        assert!((r.next() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn snapshot_equality() {
        let a = Snapshot::capture(&vec![1, 2, 3]);
        let b = Snapshot::capture(&vec![1, 2, 3]);
        let c = Snapshot::capture(&vec![1, 2, 4]);
        assert!(a.assert_eq(&b));
        assert!(!a.assert_eq(&c));
    }

    #[test]
    fn assert_within_epsilon() {
        assert!(assert_within(1.0001, 1.0, 0.001));
        assert!(!assert_within(1.01, 1.0, 0.001));
    }

    #[test]
    fn test_context_bundled() {
        let mut ctx = TestContext::new();
        ctx.timer.advance(500);
        assert_eq!(ctx.timer.now_ms(), 500);
        let r = ctx.random.next();
        assert!((r - 0.0).abs() < f64::EPSILON);
        ctx.events.record("init".into(), 0);
        assert_eq!(ctx.events.count(), 1);
    }

    #[test]
    fn wait_for_succeeds() {
        let mut counter = 0u32;
        let result = wait_for(
            || {
                counter += 1;
                counter >= 3
            },
            100,
            10,
        );
        assert!(result);
    }

    #[test]
    fn wait_for_times_out() {
        let result = wait_for(|| false, 50, 10);
        assert!(!result);
    }
}
