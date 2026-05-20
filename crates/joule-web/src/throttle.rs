//! Advanced rate control — token bucket, sliding window, leaky bucket, fixed window, adaptive.
//!
//! Replaces express-rate-limit, bottleneck, p-throttle, and rate-limiter-flexible
//! with pure-Rust rate limiting algorithms suitable for both server-side and
//! client-side usage.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ── Errors ──────────────────────────────────────────────────────

/// Rate limit errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitError {
    /// Request exceeds the rate limit.
    LimitExceeded {
        retry_after: Duration,
    },
    /// No tokens available.
    NoTokens,
}

impl std::fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LimitExceeded { retry_after } => {
                write!(f, "rate limited, retry after {:?}", retry_after)
            }
            Self::NoTokens => write!(f, "no tokens available"),
        }
    }
}

impl std::error::Error for RateLimitError {}

// ── Rate Limit Headers ─────────────────────────────────────────

/// Parsed rate limit headers (RFC 6585 / draft-ietf-httpapi-ratelimit-headers).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimitHeaders {
    /// Maximum requests allowed in the window.
    pub limit: u64,
    /// Remaining requests in the current window.
    pub remaining: u64,
    /// Seconds until the window resets.
    pub reset_seconds: u64,
    /// Optional retry-after in seconds.
    pub retry_after: Option<u64>,
}

impl RateLimitHeaders {
    /// Parse from HTTP header key-value pairs.
    pub fn from_headers(headers: &[(&str, &str)]) -> Option<Self> {
        let mut limit = None;
        let mut remaining = None;
        let mut reset = None;
        let mut retry_after = None;

        for (key, value) in headers {
            let k = key.to_ascii_lowercase();
            match k.as_str() {
                "x-ratelimit-limit" | "ratelimit-limit" => {
                    limit = value.parse().ok();
                }
                "x-ratelimit-remaining" | "ratelimit-remaining" => {
                    remaining = value.parse().ok();
                }
                "x-ratelimit-reset" | "ratelimit-reset" => {
                    reset = value.parse().ok();
                }
                "retry-after" => {
                    retry_after = value.parse().ok();
                }
                _ => {}
            }
        }

        Some(RateLimitHeaders {
            limit: limit?,
            remaining: remaining?,
            reset_seconds: reset.unwrap_or(0),
            retry_after,
        })
    }

    /// Whether the limit has been exceeded.
    pub fn is_exceeded(&self) -> bool {
        self.remaining == 0
    }
}

// ── Token Bucket ────────────────────────────────────────────────

/// Token bucket rate limiter.
///
/// Tokens refill at a steady rate. Each request consumes one token.
/// Supports burst up to the bucket capacity.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Maximum tokens (capacity).
    capacity: u64,
    /// Current token count (fractional for smooth refill).
    tokens: f64,
    /// Tokens added per second.
    refill_rate: f64,
    /// Last refill time.
    last_refill: Instant,
}

impl TokenBucket {
    pub fn new(capacity: u64, refill_rate: f64, now: Instant) -> Self {
        Self {
            capacity,
            tokens: capacity as f64,
            refill_rate,
            last_refill: now,
        }
    }

    /// Refill tokens based on elapsed time.
    fn refill(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity as f64);
        self.last_refill = now;
    }

    /// Try to consume one token.
    pub fn try_acquire(&mut self, now: Instant) -> Result<(), RateLimitError> {
        self.refill(now);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            Ok(())
        } else {
            Err(RateLimitError::NoTokens)
        }
    }

    /// Try to consume N tokens.
    pub fn try_acquire_n(&mut self, n: u64, now: Instant) -> Result<(), RateLimitError> {
        self.refill(now);
        let needed = n as f64;
        if self.tokens >= needed {
            self.tokens -= needed;
            Ok(())
        } else {
            Err(RateLimitError::NoTokens)
        }
    }

    /// Current available tokens (floored to integer).
    pub fn available(&mut self, now: Instant) -> u64 {
        self.refill(now);
        self.tokens as u64
    }

    /// Time until at least one token is available.
    pub fn time_until_available(&mut self, now: Instant) -> Duration {
        self.refill(now);
        if self.tokens >= 1.0 {
            Duration::ZERO
        } else {
            let needed = 1.0 - self.tokens;
            Duration::from_secs_f64(needed / self.refill_rate)
        }
    }
}

// ── Sliding Window ──────────────────────────────────────────────

/// Sliding window rate limiter.
///
/// Tracks exact timestamps and counts requests within the window.
#[derive(Debug, Clone)]
pub struct SlidingWindow {
    window: Duration,
    max_requests: u64,
    timestamps: VecDeque<Instant>,
}

impl SlidingWindow {
    pub fn new(window: Duration, max_requests: u64) -> Self {
        Self {
            window,
            max_requests,
            timestamps: VecDeque::new(),
        }
    }

    /// Remove expired timestamps.
    fn prune(&mut self, now: Instant) {
        while let Some(front) = self.timestamps.front() {
            if now.duration_since(*front) >= self.window {
                self.timestamps.pop_front();
            } else {
                break;
            }
        }
    }

    /// Try to record a request.
    pub fn try_acquire(&mut self, now: Instant) -> Result<(), RateLimitError> {
        self.prune(now);
        if self.timestamps.len() as u64 >= self.max_requests {
            let oldest = self.timestamps.front().unwrap();
            let retry_after = self.window - now.duration_since(*oldest);
            Err(RateLimitError::LimitExceeded { retry_after })
        } else {
            self.timestamps.push_back(now);
            Ok(())
        }
    }

    /// Current count of requests in the window.
    pub fn current_count(&mut self, now: Instant) -> u64 {
        self.prune(now);
        self.timestamps.len() as u64
    }

    /// Remaining requests allowed.
    pub fn remaining(&mut self, now: Instant) -> u64 {
        self.prune(now);
        self.max_requests.saturating_sub(self.timestamps.len() as u64)
    }
}

// ── Leaky Bucket ────────────────────────────────────────────────

/// Leaky bucket rate limiter.
///
/// Requests fill the bucket; the bucket leaks at a steady rate.
/// Overflow means rate limit exceeded.
#[derive(Debug, Clone)]
pub struct LeakyBucket {
    capacity: u64,
    /// Current water level.
    level: f64,
    /// Leak rate (units per second).
    leak_rate: f64,
    last_leak: Instant,
}

impl LeakyBucket {
    pub fn new(capacity: u64, leak_rate: f64, now: Instant) -> Self {
        Self {
            capacity,
            level: 0.0,
            leak_rate,
            last_leak: now,
        }
    }

    fn leak(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.last_leak).as_secs_f64();
        self.level = (self.level - elapsed * self.leak_rate).max(0.0);
        self.last_leak = now;
    }

    /// Try to add a request to the bucket.
    pub fn try_acquire(&mut self, now: Instant) -> Result<(), RateLimitError> {
        self.leak(now);
        if self.level + 1.0 <= self.capacity as f64 {
            self.level += 1.0;
            Ok(())
        } else {
            let overflow = self.level + 1.0 - self.capacity as f64;
            let retry_after = Duration::from_secs_f64(overflow / self.leak_rate);
            Err(RateLimitError::LimitExceeded { retry_after })
        }
    }

    /// Current water level.
    pub fn current_level(&mut self, now: Instant) -> f64 {
        self.leak(now);
        self.level
    }
}

// ── Fixed Window Counter ────────────────────────────────────────

/// Fixed window counter rate limiter.
///
/// Counts requests in fixed time windows. Simpler but less smooth than sliding.
#[derive(Debug, Clone)]
pub struct FixedWindowCounter {
    window: Duration,
    max_requests: u64,
    count: u64,
    window_start: Instant,
}

impl FixedWindowCounter {
    pub fn new(window: Duration, max_requests: u64, now: Instant) -> Self {
        Self {
            window,
            max_requests,
            count: 0,
            window_start: now,
        }
    }

    fn maybe_reset(&mut self, now: Instant) {
        if now.duration_since(self.window_start) >= self.window {
            self.count = 0;
            self.window_start = now;
        }
    }

    pub fn try_acquire(&mut self, now: Instant) -> Result<(), RateLimitError> {
        self.maybe_reset(now);
        if self.count < self.max_requests {
            self.count += 1;
            Ok(())
        } else {
            let elapsed = now.duration_since(self.window_start);
            let retry_after = self.window.saturating_sub(elapsed);
            Err(RateLimitError::LimitExceeded { retry_after })
        }
    }

    pub fn remaining(&mut self, now: Instant) -> u64 {
        self.maybe_reset(now);
        self.max_requests.saturating_sub(self.count)
    }

    pub fn count(&mut self, now: Instant) -> u64 {
        self.maybe_reset(now);
        self.count
    }
}

// ── Adaptive Rate Limiter ───────────────────────────────────────

/// Adaptive rate limiter that adjusts limits based on success/failure signals.
#[derive(Debug, Clone)]
pub struct AdaptiveRateLimiter {
    /// Current effective limit.
    current_limit: u64,
    /// Minimum limit.
    min_limit: u64,
    /// Maximum limit.
    max_limit: u64,
    /// Increase step on success.
    increase_step: u64,
    /// Decrease factor on failure (multiply current by this, e.g. 0.5).
    decrease_factor: f64,
    /// Underlying fixed window counter.
    counter: FixedWindowCounter,
}

impl AdaptiveRateLimiter {
    pub fn new(
        initial_limit: u64,
        min_limit: u64,
        max_limit: u64,
        window: Duration,
        now: Instant,
    ) -> Self {
        Self {
            current_limit: initial_limit,
            min_limit,
            max_limit,
            increase_step: 1,
            decrease_factor: 0.5,
            counter: FixedWindowCounter::new(window, initial_limit, now),
        }
    }

    pub fn try_acquire(&mut self, now: Instant) -> Result<(), RateLimitError> {
        self.counter.max_requests = self.current_limit;
        self.counter.try_acquire(now)
    }

    /// Signal success — increase the limit.
    pub fn on_success(&mut self) {
        self.current_limit =
            (self.current_limit + self.increase_step).min(self.max_limit);
        self.counter.max_requests = self.current_limit;
    }

    /// Signal failure — decrease the limit.
    pub fn on_failure(&mut self) {
        let new_limit = (self.current_limit as f64 * self.decrease_factor) as u64;
        self.current_limit = new_limit.max(self.min_limit);
        self.counter.max_requests = self.current_limit;
    }

    pub fn current_limit(&self) -> u64 {
        self.current_limit
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    fn sec(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    #[test]
    fn token_bucket_basic() {
        let start = Instant::now();
        let mut tb = TokenBucket::new(5, 1.0, start);

        // Should have 5 tokens initially.
        assert_eq!(tb.available(start), 5);

        // Consume 3.
        for _ in 0..3 {
            assert!(tb.try_acquire(start).is_ok());
        }
        assert_eq!(tb.available(start), 2);
    }

    #[test]
    fn token_bucket_refill() {
        let start = Instant::now();
        let mut tb = TokenBucket::new(5, 10.0, start); // 10 tokens/sec

        // Drain all tokens.
        for _ in 0..5 {
            tb.try_acquire(start).unwrap();
        }
        assert!(tb.try_acquire(start).is_err());

        // After 0.5 seconds, should have ~5 tokens.
        assert!(tb.try_acquire(start + ms(500)).is_ok());
    }

    #[test]
    fn token_bucket_burst() {
        let start = Instant::now();
        let mut tb = TokenBucket::new(10, 1.0, start);
        assert!(tb.try_acquire_n(10, start).is_ok());
        assert!(tb.try_acquire(start).is_err());
    }

    #[test]
    fn sliding_window_basic() {
        let start = Instant::now();
        let mut sw = SlidingWindow::new(sec(1), 3);

        assert!(sw.try_acquire(start).is_ok());
        assert!(sw.try_acquire(start + ms(100)).is_ok());
        assert!(sw.try_acquire(start + ms(200)).is_ok());
        assert!(sw.try_acquire(start + ms(300)).is_err());

        // After window expires, should allow again.
        assert!(sw.try_acquire(start + sec(1) + ms(100)).is_ok());
    }

    #[test]
    fn sliding_window_remaining() {
        let start = Instant::now();
        let mut sw = SlidingWindow::new(sec(1), 5);
        sw.try_acquire(start).unwrap();
        sw.try_acquire(start + ms(10)).unwrap();
        assert_eq!(sw.remaining(start + ms(20)), 3);
    }

    #[test]
    fn leaky_bucket_basic() {
        let start = Instant::now();
        let mut lb = LeakyBucket::new(3, 1.0, start);

        assert!(lb.try_acquire(start).is_ok());
        assert!(lb.try_acquire(start).is_ok());
        assert!(lb.try_acquire(start).is_ok());
        assert!(lb.try_acquire(start).is_err()); // Overflow.

        // After 1 second, 1 unit leaks.
        assert!(lb.try_acquire(start + sec(1)).is_ok());
    }

    #[test]
    fn fixed_window_counter() {
        let start = Instant::now();
        let mut fw = FixedWindowCounter::new(sec(1), 3, start);

        assert!(fw.try_acquire(start).is_ok());
        assert!(fw.try_acquire(start + ms(100)).is_ok());
        assert!(fw.try_acquire(start + ms(200)).is_ok());
        assert!(fw.try_acquire(start + ms(300)).is_err());
        assert_eq!(fw.remaining(start + ms(400)), 0);

        // Next window.
        assert!(fw.try_acquire(start + sec(1)).is_ok());
        assert_eq!(fw.count(start + sec(1)), 1);
    }

    #[test]
    fn adaptive_rate_limiter() {
        let start = Instant::now();
        let mut ar = AdaptiveRateLimiter::new(5, 1, 20, sec(1), start);

        assert_eq!(ar.current_limit(), 5);

        // Success increases limit.
        ar.on_success();
        assert_eq!(ar.current_limit(), 6);

        // Failure decreases limit.
        ar.on_failure();
        assert_eq!(ar.current_limit(), 3); // 6 * 0.5 = 3
    }

    #[test]
    fn adaptive_respects_bounds() {
        let start = Instant::now();
        let mut ar = AdaptiveRateLimiter::new(2, 2, 3, sec(1), start);

        ar.on_failure();
        assert_eq!(ar.current_limit(), 2); // Can't go below min.

        ar.on_success();
        ar.on_success();
        ar.on_success();
        assert_eq!(ar.current_limit(), 3); // Can't go above max.
    }

    #[test]
    fn rate_limit_headers_parse() {
        let headers = vec![
            ("X-RateLimit-Limit", "100"),
            ("X-RateLimit-Remaining", "42"),
            ("X-RateLimit-Reset", "30"),
            ("Retry-After", "5"),
        ];

        let parsed = RateLimitHeaders::from_headers(&headers).unwrap();
        assert_eq!(parsed.limit, 100);
        assert_eq!(parsed.remaining, 42);
        assert_eq!(parsed.reset_seconds, 30);
        assert_eq!(parsed.retry_after, Some(5));
        assert!(!parsed.is_exceeded());
    }

    #[test]
    fn rate_limit_headers_exceeded() {
        let headers = vec![
            ("ratelimit-limit", "10"),
            ("ratelimit-remaining", "0"),
        ];
        let parsed = RateLimitHeaders::from_headers(&headers).unwrap();
        assert!(parsed.is_exceeded());
    }

    #[test]
    fn token_bucket_time_until_available() {
        let start = Instant::now();
        let mut tb = TokenBucket::new(1, 2.0, start); // 2 tokens/sec
        tb.try_acquire(start).unwrap();
        let wait = tb.time_until_available(start);
        // Should need ~0.5 seconds.
        assert!(wait.as_millis() > 400 && wait.as_millis() < 600);
    }

    #[test]
    fn leaky_bucket_level() {
        let start = Instant::now();
        let mut lb = LeakyBucket::new(10, 2.0, start);
        lb.try_acquire(start).unwrap();
        lb.try_acquire(start).unwrap();
        // Level should be close to 2.0 at start.
        let level = lb.current_level(start);
        assert!((level - 2.0).abs() < 0.01);

        // After 1 second, should have leaked 2 units.
        let level = lb.current_level(start + sec(1));
        assert!(level < 0.01);
    }
}
