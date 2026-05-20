//! Client-side rate limiting utilities.
//!
//! Provides multiple rate-limiting strategies for controlling request
//! frequency in web applications without external dependencies:
//!
//! - [`RateLimiter`] — fixed-window sliding log
//! - [`TokenBucket`] — classic token bucket with continuous refill
//! - [`SlidingWindowCounter`] — approximate sliding window via sub-buckets
//! - [`Debouncer`] — suppress rapid-fire calls, emit only after quiet period
//! - [`Throttle`] — allow at most one call per interval

use std::collections::VecDeque;

// ── Fixed-Window Sliding Log ────────────────────────────────────────

/// Tracks individual request timestamps inside a sliding time window.
/// Requests are allowed only when the count of timestamps within the
/// most recent `window_ms` is below `max_requests`.
pub struct RateLimiter {
    max_requests: u32,
    window_ms: u64,
    requests: VecDeque<u64>,
}

impl RateLimiter {
    pub fn new(max_requests: u32, window_ms: u64) -> Self {
        Self {
            max_requests,
            window_ms,
            requests: VecDeque::new(),
        }
    }

    /// Attempt to acquire a permit at time `now_ms`.
    /// Returns `true` if the request is allowed, `false` if rate-limited.
    pub fn try_acquire(&mut self, now_ms: u64) -> bool {
        self.prune(now_ms);
        if (self.requests.len() as u32) < self.max_requests {
            self.requests.push_back(now_ms);
            true
        } else {
            false
        }
    }

    /// Number of requests still available in the current window.
    pub fn remaining(&self, now_ms: u64) -> u32 {
        let active = self.count_active(now_ms);
        self.max_requests.saturating_sub(active)
    }

    /// Milliseconds until the oldest active request falls out of the window,
    /// freeing one slot.  Returns 0 when there is already capacity.
    pub fn reset_after_ms(&self, now_ms: u64) -> u64 {
        if self.remaining(now_ms) > 0 {
            return 0;
        }
        // Oldest request still inside the window
        for ts in &self.requests {
            if *ts + self.window_ms > now_ms {
                return (*ts + self.window_ms).saturating_sub(now_ms);
            }
        }
        0
    }

    // ── internal helpers ──

    fn prune(&mut self, now_ms: u64) {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        while let Some(&front) = self.requests.front() {
            if front < cutoff {
                self.requests.pop_front();
            } else {
                break;
            }
        }
    }

    fn count_active(&self, now_ms: u64) -> u32 {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        self.requests.iter().filter(|&&ts| ts >= cutoff).count() as u32
    }
}

// ── Token Bucket ────────────────────────────────────────────────────

/// Classic token-bucket rate limiter with continuous refill.
pub struct TokenBucket {
    capacity: u32,
    tokens: f64,
    refill_rate: f64, // tokens per millisecond
    last_refill_ms: u64,
}

impl TokenBucket {
    /// Create a bucket that holds `capacity` tokens and refills at
    /// `refill_rate_per_sec` tokens per second.  Starts full.
    pub fn new(capacity: u32, refill_rate_per_sec: f64) -> Self {
        Self {
            capacity,
            tokens: capacity as f64,
            refill_rate: refill_rate_per_sec / 1000.0,
            last_refill_ms: 0,
        }
    }

    /// Try to consume one token at time `now_ms`.
    pub fn try_acquire(&mut self, now_ms: u64) -> bool {
        self.try_acquire_n(1, now_ms)
    }

    /// Try to consume `n` tokens at time `now_ms`.
    pub fn try_acquire_n(&mut self, n: u32, now_ms: u64) -> bool {
        self.refill(now_ms);
        let needed = n as f64;
        if self.tokens >= needed {
            self.tokens -= needed;
            true
        } else {
            false
        }
    }

    /// Tokens available right now (after refilling to `now_ms`).
    pub fn available_tokens(&self, now_ms: u64) -> u32 {
        let elapsed = now_ms.saturating_sub(self.last_refill_ms) as f64;
        let t = (self.tokens + elapsed * self.refill_rate).min(self.capacity as f64);
        t as u32
    }

    fn refill(&mut self, now_ms: u64) {
        let elapsed = now_ms.saturating_sub(self.last_refill_ms) as f64;
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity as f64);
        self.last_refill_ms = now_ms;
    }
}

// ── Sliding Window Counter ──────────────────────────────────────────

/// Approximate sliding-window counter using fixed sub-buckets.
pub struct SlidingWindowCounter {
    window_ms: u64,
    buckets: Vec<(u64, u32)>, // (bucket_start_ms, count)
    max_requests: u32,
    bucket_count: usize,
}

impl SlidingWindowCounter {
    pub fn new(max_requests: u32, window_ms: u64, bucket_count: usize) -> Self {
        let bucket_count = bucket_count.max(1);
        Self {
            window_ms,
            buckets: Vec::with_capacity(bucket_count),
            max_requests,
            bucket_count,
        }
    }

    /// Record an event at `now_ms`.  Returns `true` if allowed.
    pub fn record(&mut self, now_ms: u64) -> bool {
        self.prune(now_ms);
        if self.count(now_ms) >= self.max_requests {
            return false;
        }
        let bucket_width = self.window_ms / self.bucket_count as u64;
        let bucket_start = if bucket_width > 0 {
            (now_ms / bucket_width) * bucket_width
        } else {
            now_ms
        };
        if let Some(last) = self.buckets.last_mut() {
            if last.0 == bucket_start {
                last.1 += 1;
                return true;
            }
        }
        self.buckets.push((bucket_start, 1));
        true
    }

    /// Current count of events inside the sliding window.
    pub fn count(&self, now_ms: u64) -> u32 {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        self.buckets
            .iter()
            .filter(|(start, _)| *start >= cutoff)
            .map(|(_, c)| *c)
            .sum()
    }

    fn prune(&mut self, now_ms: u64) {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        self.buckets.retain(|(start, _)| *start >= cutoff);
    }
}

// ── Debouncer ───────────────────────────────────────────────────────

/// Suppresses rapid-fire calls; returns `true` only after `delay_ms`
/// of silence since the last call.
pub struct Debouncer {
    delay_ms: u64,
    last_call_ms: Option<u64>,
    pending: bool,
}

impl Debouncer {
    pub fn new(delay_ms: u64) -> Self {
        Self {
            delay_ms,
            last_call_ms: None,
            pending: false,
        }
    }

    /// Signal a call at `now_ms`.  Returns `true` if the debounce
    /// delay has elapsed since the last call (or if this is the first
    /// call after a quiet period).
    pub fn call(&mut self, now_ms: u64) -> bool {
        match self.last_call_ms {
            None => {
                self.last_call_ms = Some(now_ms);
                self.pending = true;
                // First call is allowed immediately
                true
            }
            Some(prev) => {
                if now_ms.saturating_sub(prev) >= self.delay_ms {
                    self.last_call_ms = Some(now_ms);
                    self.pending = false;
                    true
                } else {
                    self.last_call_ms = Some(now_ms);
                    self.pending = true;
                    false
                }
            }
        }
    }

    /// Whether there is a pending call that hasn't fired yet.
    pub fn is_pending(&self) -> bool {
        self.pending
    }
}

// ── Throttle ────────────────────────────────────────────────────────

/// Allows at most one call per `interval_ms`.
pub struct Throttle {
    interval_ms: u64,
    last_allowed_ms: Option<u64>,
}

impl Throttle {
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval_ms,
            last_allowed_ms: None,
        }
    }

    /// Returns `true` if the call is allowed (at most once per interval).
    pub fn try_call(&mut self, now_ms: u64) -> bool {
        match self.last_allowed_ms {
            None => {
                self.last_allowed_ms = Some(now_ms);
                true
            }
            Some(prev) => {
                if now_ms.saturating_sub(prev) >= self.interval_ms {
                    self.last_allowed_ms = Some(now_ms);
                    true
                } else {
                    false
                }
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RateLimiter ─────────────────────────────────────────────

    #[test]
    fn rate_limiter_allows_up_to_max() {
        let mut rl = RateLimiter::new(3, 1000);
        assert!(rl.try_acquire(100));
        assert!(rl.try_acquire(200));
        assert!(rl.try_acquire(300));
    }

    #[test]
    fn rate_limiter_blocks_after_max() {
        let mut rl = RateLimiter::new(2, 1000);
        assert!(rl.try_acquire(100));
        assert!(rl.try_acquire(200));
        assert!(!rl.try_acquire(300));
    }

    #[test]
    fn rate_limiter_window_expires() {
        let mut rl = RateLimiter::new(2, 1000);
        assert!(rl.try_acquire(0));
        assert!(rl.try_acquire(100));
        assert!(!rl.try_acquire(500));
        // After the window slides past the first request
        assert!(rl.try_acquire(1001));
    }

    #[test]
    fn rate_limiter_remaining_count() {
        let mut rl = RateLimiter::new(5, 1000);
        assert_eq!(rl.remaining(0), 5);
        rl.try_acquire(100);
        rl.try_acquire(200);
        assert_eq!(rl.remaining(300), 3);
    }

    #[test]
    fn rate_limiter_reset_timing() {
        let mut rl = RateLimiter::new(2, 1000);
        rl.try_acquire(0);
        rl.try_acquire(100);
        // Full — reset_after_ms tells us when the oldest expires
        let reset = rl.reset_after_ms(500);
        assert_eq!(reset, 500); // 0 + 1000 - 500 = 500
    }

    // ── TokenBucket ─────────────────────────────────────────────

    #[test]
    fn token_bucket_refills() {
        let mut tb = TokenBucket::new(10, 10.0); // 10 tokens/sec
        // Drain all tokens
        for _ in 0..10 {
            assert!(tb.try_acquire(0));
        }
        assert!(!tb.try_acquire(0));
        // After 500ms, should have ~5 tokens
        assert!(tb.try_acquire(500));
    }

    #[test]
    fn token_bucket_empty_blocks() {
        let mut tb = TokenBucket::new(2, 1.0);
        assert!(tb.try_acquire(0));
        assert!(tb.try_acquire(0));
        assert!(!tb.try_acquire(0));
    }

    #[test]
    fn token_bucket_acquire_n() {
        let mut tb = TokenBucket::new(10, 10.0);
        assert!(tb.try_acquire_n(5, 0));
        assert!(tb.try_acquire_n(5, 0));
        assert!(!tb.try_acquire_n(1, 0));
    }

    #[test]
    fn token_bucket_available_tokens() {
        let mut tb = TokenBucket::new(10, 10.0);
        for _ in 0..10 {
            tb.try_acquire(0);
        }
        // After 1 second, should be back to 10
        assert_eq!(tb.available_tokens(1000), 10);
    }

    // ── SlidingWindowCounter ────────────────────────────────────

    #[test]
    fn sliding_window_allows_and_blocks() {
        let mut sw = SlidingWindowCounter::new(3, 1000, 4);
        assert!(sw.record(100));
        assert!(sw.record(200));
        assert!(sw.record(300));
        assert!(!sw.record(400));
        // After window passes
        assert!(sw.record(1200));
    }

    #[test]
    fn sliding_window_count() {
        let mut sw = SlidingWindowCounter::new(100, 1000, 4);
        sw.record(100);
        sw.record(200);
        sw.record(900);
        assert_eq!(sw.count(950), 3);
        // First two should expire
        assert_eq!(sw.count(1200), 1);
    }

    // ── Debouncer ───────────────────────────────────────────────

    #[test]
    fn debouncer_delays() {
        let mut d = Debouncer::new(200);
        assert!(d.call(0));      // first call fires immediately
        assert!(!d.call(50));    // too soon
        assert!(!d.call(100));   // still too soon
        assert!(d.call(300));    // 200ms since last call (at 100)
    }

    #[test]
    fn debouncer_pending() {
        let mut d = Debouncer::new(100);
        d.call(0);
        d.call(50); // suppressed
        assert!(d.is_pending());
    }

    // ── Throttle ────────────────────────────────────────────────

    #[test]
    fn throttle_limits_frequency() {
        let mut t = Throttle::new(100);
        assert!(t.try_call(0));
        assert!(!t.try_call(50));
        assert!(t.try_call(100));
        assert!(!t.try_call(150));
        assert!(t.try_call(200));
    }

    #[test]
    fn throttle_first_call_always_allowed() {
        let mut t = Throttle::new(500);
        assert!(t.try_call(1234));
    }
}
