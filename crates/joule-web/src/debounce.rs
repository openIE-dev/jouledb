//! Debounce/throttle utilities — configurable delay, leading/trailing edge, cancel/flush.
//!
//! Replaces lodash.debounce, lodash.throttle, and use-debounce with pure-Rust
//! debounce and throttle primitives using std::time instants.

use std::time::{Duration, Instant};

// ── Debounce ────────────────────────────────────────────────────

/// Edge on which to fire the debounced call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// Fire on the leading edge (immediately, then suppress).
    Leading,
    /// Fire on the trailing edge (after quiet period).
    Trailing,
    /// Fire on both edges.
    Both,
}

/// Debounce state machine.
///
/// Tracks timing without spawning threads. Call `call()` when the event occurs
/// and `poll()` to check if the debounced action should fire.
#[derive(Debug, Clone)]
pub struct Debouncer {
    delay: Duration,
    max_wait: Option<Duration>,
    edge: Edge,
    /// When the last call() happened.
    last_call: Option<Instant>,
    /// When the first call in the current burst happened.
    burst_start: Option<Instant>,
    /// Whether the leading edge has fired for the current burst.
    leading_fired: bool,
    /// Whether a trailing fire is pending.
    trailing_pending: bool,
    /// Number of calls suppressed.
    suppressed_count: u64,
    /// Whether the debouncer has been cancelled.
    cancelled: bool,
}

impl Debouncer {
    pub fn new(delay: Duration) -> Self {
        Self {
            delay,
            max_wait: None,
            edge: Edge::Trailing,
            last_call: None,
            burst_start: None,
            leading_fired: false,
            trailing_pending: false,
            suppressed_count: 0,
            cancelled: false,
        }
    }

    pub fn with_edge(mut self, edge: Edge) -> Self {
        self.edge = edge;
        self
    }

    pub fn with_max_wait(mut self, max: Duration) -> Self {
        self.max_wait = Some(max);
        self
    }

    /// Record a call event. Returns `true` if the action should fire immediately
    /// (leading edge).
    pub fn call(&mut self, now: Instant) -> bool {
        self.cancelled = false;
        self.last_call = Some(now);

        if self.burst_start.is_none() {
            self.burst_start = Some(now);
        }

        let should_fire_leading = match self.edge {
            Edge::Leading | Edge::Both => !self.leading_fired,
            Edge::Trailing => false,
        };

        if should_fire_leading {
            self.leading_fired = true;
            self.trailing_pending = false;
            return true;
        }

        self.suppressed_count += 1;
        self.trailing_pending = true;
        false
    }

    /// Check if the trailing action should fire. Returns `true` if enough time
    /// has passed since the last call.
    pub fn poll(&mut self, now: Instant) -> bool {
        if self.cancelled {
            return false;
        }

        // Check max_wait first.
        if let (Some(max_wait), Some(burst_start)) = (self.max_wait, self.burst_start) {
            if now.duration_since(burst_start) >= max_wait && self.trailing_pending {
                self.reset();
                return true;
            }
        }

        // Check trailing edge.
        if let Some(last) = self.last_call {
            if now.duration_since(last) >= self.delay && self.trailing_pending {
                match self.edge {
                    Edge::Trailing | Edge::Both => {
                        self.reset();
                        return true;
                    }
                    Edge::Leading => {
                        self.reset();
                        return false;
                    }
                }
            }
        }

        false
    }

    /// Cancel any pending debounced call.
    pub fn cancel(&mut self) {
        self.cancelled = true;
        self.trailing_pending = false;
        self.reset();
    }

    /// Flush: if there's a pending trailing call, fire it now.
    pub fn flush(&mut self) -> bool {
        if self.trailing_pending && !self.cancelled {
            self.reset();
            return true;
        }
        false
    }

    /// Number of suppressed calls in the current burst.
    pub fn suppressed(&self) -> u64 {
        self.suppressed_count
    }

    /// Whether a trailing call is pending.
    pub fn is_pending(&self) -> bool {
        self.trailing_pending && !self.cancelled
    }

    fn reset(&mut self) {
        self.leading_fired = false;
        self.trailing_pending = false;
        self.burst_start = None;
        self.suppressed_count = 0;
    }
}

// ── Simple Throttle ─────────────────────────────────────────────

/// Simple throttle that limits calls to at most one per interval.
#[derive(Debug, Clone)]
pub struct Throttle {
    interval: Duration,
    last_fired: Option<Instant>,
    /// Whether to fire a trailing call after the interval.
    trailing: bool,
    trailing_pending: bool,
}

impl Throttle {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_fired: None,
            trailing: false,
            trailing_pending: false,
        }
    }

    pub fn with_trailing(mut self, trailing: bool) -> Self {
        self.trailing = trailing;
        self
    }

    /// Attempt to fire. Returns `true` if the action should execute.
    pub fn call(&mut self, now: Instant) -> bool {
        match self.last_fired {
            None => {
                self.last_fired = Some(now);
                true
            }
            Some(last) => {
                if now.duration_since(last) >= self.interval {
                    self.last_fired = Some(now);
                    self.trailing_pending = false;
                    true
                } else {
                    if self.trailing {
                        self.trailing_pending = true;
                    }
                    false
                }
            }
        }
    }

    /// Poll for trailing call.
    pub fn poll(&mut self, now: Instant) -> bool {
        if !self.trailing || !self.trailing_pending {
            return false;
        }
        if let Some(last) = self.last_fired {
            if now.duration_since(last) >= self.interval {
                self.last_fired = Some(now);
                self.trailing_pending = false;
                return true;
            }
        }
        false
    }

    /// Reset the throttle.
    pub fn reset(&mut self) {
        self.last_fired = None;
        self.trailing_pending = false;
    }

    /// Time remaining until next allowed call.
    pub fn remaining(&self, now: Instant) -> Duration {
        match self.last_fired {
            None => Duration::ZERO,
            Some(last) => {
                let elapsed = now.duration_since(last);
                if elapsed >= self.interval {
                    Duration::ZERO
                } else {
                    self.interval - elapsed
                }
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn trailing_debounce_basic() {
        let start = Instant::now();
        let mut d = Debouncer::new(ms(100));

        // Call should not fire immediately (trailing edge).
        assert!(!d.call(start));
        assert!(d.is_pending());

        // Before delay: should not fire.
        assert!(!d.poll(start + ms(50)));

        // After delay: should fire.
        assert!(d.poll(start + ms(100)));
        assert!(!d.is_pending());
    }

    #[test]
    fn leading_debounce() {
        let start = Instant::now();
        let mut d = Debouncer::new(ms(100)).with_edge(Edge::Leading);

        // First call fires immediately.
        assert!(d.call(start));
        // Subsequent calls are suppressed.
        assert!(!d.call(start + ms(10)));
        assert!(!d.call(start + ms(20)));
    }

    #[test]
    fn both_edges() {
        let start = Instant::now();
        let mut d = Debouncer::new(ms(100)).with_edge(Edge::Both);

        // Leading fires.
        assert!(d.call(start));
        // More calls suppressed.
        assert!(!d.call(start + ms(10)));
        // After delay, trailing fires.
        assert!(d.poll(start + ms(110)));
    }

    #[test]
    fn cancel_debounce() {
        let start = Instant::now();
        let mut d = Debouncer::new(ms(100));
        d.call(start);
        d.cancel();
        assert!(!d.poll(start + ms(200)));
        assert!(!d.is_pending());
    }

    #[test]
    fn flush_debounce() {
        let start = Instant::now();
        let mut d = Debouncer::new(ms(100));
        d.call(start);
        assert!(d.flush());
        assert!(!d.is_pending());
    }

    #[test]
    fn flush_empty() {
        let mut d = Debouncer::new(ms(100));
        assert!(!d.flush()); // nothing pending
    }

    #[test]
    fn max_wait() {
        let start = Instant::now();
        let mut d = Debouncer::new(ms(100)).with_max_wait(ms(200));

        d.call(start);
        // Keep calling to reset the delay timer.
        d.call(start + ms(80));
        d.call(start + ms(160));

        // At 200ms from burst start, should fire even though last call was recent.
        assert!(d.poll(start + ms(200)));
    }

    #[test]
    fn suppressed_count() {
        let start = Instant::now();
        let mut d = Debouncer::new(ms(100));
        d.call(start);
        d.call(start + ms(10));
        d.call(start + ms(20));
        // All 3 calls are suppressed (trailing edge, none fire immediately).
        assert_eq!(d.suppressed(), 3);
    }

    #[test]
    fn throttle_basic() {
        let start = Instant::now();
        let mut t = Throttle::new(ms(100));

        assert!(t.call(start)); // First call allowed.
        assert!(!t.call(start + ms(50))); // Too soon.
        assert!(t.call(start + ms(100))); // Enough time passed.
    }

    #[test]
    fn throttle_with_trailing() {
        let start = Instant::now();
        let mut t = Throttle::new(ms(100)).with_trailing(true);

        assert!(t.call(start));
        assert!(!t.call(start + ms(50))); // Suppressed, but trailing pending.
        assert!(!t.poll(start + ms(50))); // Not yet.
        assert!(t.poll(start + ms(100))); // Trailing fires.
    }

    #[test]
    fn throttle_remaining() {
        let start = Instant::now();
        let mut t = Throttle::new(ms(100));
        t.call(start);
        assert_eq!(t.remaining(start + ms(30)), ms(70));
        assert_eq!(t.remaining(start + ms(100)), ms(0));
    }

    #[test]
    fn throttle_reset() {
        let start = Instant::now();
        let mut t = Throttle::new(ms(100));
        t.call(start);
        t.reset();
        assert!(t.call(start + ms(10))); // Should be allowed after reset.
    }

    #[test]
    fn debouncer_reuse_after_fire() {
        let start = Instant::now();
        let mut d = Debouncer::new(ms(50));

        d.call(start);
        assert!(d.poll(start + ms(50)));

        // New burst.
        assert!(!d.call(start + ms(100)));
        assert!(d.poll(start + ms(150)));
    }
}
