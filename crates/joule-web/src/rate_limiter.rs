//! Rate limiting algorithms — token bucket, sliding window, fixed window, leaky bucket.
//!
//! Replaces express-rate-limit, bottleneck, and limiter with pure-Rust rate
//! limiting that supports per-key tracking, burst allowance, configurable time
//! windows, and rate limit response headers (X-RateLimit-*).

use std::collections::HashMap;

// ── Errors ─────────────────────────────────────────────────────

/// Rate limiter errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimiterError {
    /// Rate limit exceeded.
    LimitExceeded { retry_after_ms: u64 },
    /// Invalid configuration.
    InvalidConfig(String),
    /// Key not found.
    KeyNotFound(String),
}

impl std::fmt::Display for RateLimiterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LimitExceeded { retry_after_ms } => {
                write!(f, "rate limit exceeded, retry after {retry_after_ms}ms")
            }
            Self::InvalidConfig(s) => write!(f, "invalid rate limiter config: {s}"),
            Self::KeyNotFound(k) => write!(f, "key not found: {k}"),
        }
    }
}

impl std::error::Error for RateLimiterError {}

// ── Rate Limit Headers ─────────────────────────────────────────

/// Standard rate limit response headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateLimitHeaders {
    /// Maximum requests allowed in the window.
    pub limit: u64,
    /// Remaining requests in the current window.
    pub remaining: u64,
    /// Unix timestamp (seconds) when the window resets.
    pub reset: u64,
    /// Milliseconds until a retry is allowed (only set when limited).
    pub retry_after_ms: Option<u64>,
}

impl RateLimitHeaders {
    /// Render as HTTP header key-value pairs.
    pub fn to_header_pairs(&self) -> Vec<(String, String)> {
        let mut headers = vec![
            ("X-RateLimit-Limit".to_string(), self.limit.to_string()),
            (
                "X-RateLimit-Remaining".to_string(),
                self.remaining.to_string(),
            ),
            ("X-RateLimit-Reset".to_string(), self.reset.to_string()),
        ];
        if let Some(retry) = self.retry_after_ms {
            let secs = (retry + 999) / 1000; // ceil to seconds
            headers.push(("Retry-After".to_string(), secs.to_string()));
        }
        headers
    }
}

// ── Decision ────────────────────────────────────────────────────

/// Result of a rate limit check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateLimitDecision {
    /// Whether the request is allowed.
    pub allowed: bool,
    /// Response headers to include.
    pub headers: RateLimitHeaders,
}

// ── Token Bucket ────────────────────────────────────────────────

/// Token bucket rate limiter with continuous refill.
///
/// Allows bursts up to `capacity`, refilling at `refill_rate` tokens per second.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Maximum tokens (burst capacity).
    capacity: f64,
    /// Tokens added per second.
    refill_rate: f64,
    /// Per-key state.
    buckets: HashMap<String, TokenBucketState>,
}

#[derive(Debug, Clone)]
struct TokenBucketState {
    tokens: f64,
    last_refill_ms: u64,
}

impl TokenBucket {
    /// Create a new token bucket limiter.
    ///
    /// `capacity` is the burst size; `refill_rate` is tokens added per second.
    pub fn new(capacity: f64, refill_rate: f64) -> Result<Self, RateLimiterError> {
        if capacity <= 0.0 {
            return Err(RateLimiterError::InvalidConfig(
                "capacity must be > 0".to_string(),
            ));
        }
        if refill_rate <= 0.0 {
            return Err(RateLimiterError::InvalidConfig(
                "refill_rate must be > 0".to_string(),
            ));
        }
        Ok(Self {
            capacity,
            refill_rate,
            buckets: HashMap::new(),
        })
    }

    /// Check whether a request for `key` is allowed at time `now_ms`.
    pub fn check(&mut self, key: &str, now_ms: u64) -> RateLimitDecision {
        self.try_consume(key, 1.0, now_ms)
    }

    /// Try to consume `tokens` from the bucket for `key`.
    pub fn try_consume(&mut self, key: &str, tokens: f64, now_ms: u64) -> RateLimitDecision {
        let state = self
            .buckets
            .entry(key.to_string())
            .or_insert(TokenBucketState {
                tokens: self.capacity,
                last_refill_ms: now_ms,
            });

        // Refill tokens based on elapsed time.
        let elapsed_ms = now_ms.saturating_sub(state.last_refill_ms);
        let refill = (elapsed_ms as f64 / 1000.0) * self.refill_rate;
        state.tokens = (state.tokens + refill).min(self.capacity);
        state.last_refill_ms = now_ms;

        if state.tokens >= tokens {
            state.tokens -= tokens;
            let remaining = state.tokens as u64;
            let reset_secs = now_ms / 1000 + 1;
            RateLimitDecision {
                allowed: true,
                headers: RateLimitHeaders {
                    limit: self.capacity as u64,
                    remaining,
                    reset: reset_secs,
                    retry_after_ms: None,
                },
            }
        } else {
            // How long until we have enough tokens?
            let deficit = tokens - state.tokens;
            let wait_ms = ((deficit / self.refill_rate) * 1000.0).ceil() as u64;
            let reset_secs = (now_ms + wait_ms) / 1000 + 1;
            RateLimitDecision {
                allowed: false,
                headers: RateLimitHeaders {
                    limit: self.capacity as u64,
                    remaining: 0,
                    reset: reset_secs,
                    retry_after_ms: Some(wait_ms),
                },
            }
        }
    }

    /// Get current token count for a key.
    pub fn tokens_remaining(&self, key: &str) -> Option<f64> {
        self.buckets.get(key).map(|s| s.tokens)
    }

    /// Remove tracking for a key.
    pub fn remove_key(&mut self, key: &str) {
        self.buckets.remove(key);
    }

    /// Number of tracked keys.
    pub fn key_count(&self) -> usize {
        self.buckets.len()
    }
}

// ── Sliding Window Log ──────────────────────────────────────────

/// Sliding window rate limiter using a request log.
///
/// Keeps exact timestamps of each request and counts those within
/// the window. Precise but uses more memory.
#[derive(Debug, Clone)]
pub struct SlidingWindowLog {
    /// Maximum requests per window.
    max_requests: u64,
    /// Window duration in milliseconds.
    window_ms: u64,
    /// Per-key request timestamps.
    logs: HashMap<String, Vec<u64>>,
}

impl SlidingWindowLog {
    /// Create a new sliding window limiter.
    pub fn new(max_requests: u64, window_ms: u64) -> Result<Self, RateLimiterError> {
        if max_requests == 0 {
            return Err(RateLimiterError::InvalidConfig(
                "max_requests must be > 0".to_string(),
            ));
        }
        if window_ms == 0 {
            return Err(RateLimiterError::InvalidConfig(
                "window_ms must be > 0".to_string(),
            ));
        }
        Ok(Self {
            max_requests,
            window_ms,
            logs: HashMap::new(),
        })
    }

    /// Check and record a request for `key` at time `now_ms`.
    pub fn check(&mut self, key: &str, now_ms: u64) -> RateLimitDecision {
        let log = self.logs.entry(key.to_string()).or_default();

        // Remove entries outside the window.
        let window_start = now_ms.saturating_sub(self.window_ms);
        log.retain(|ts| *ts > window_start);

        let count = log.len() as u64;
        let reset_secs = (now_ms + self.window_ms) / 1000;

        if count < self.max_requests {
            log.push(now_ms);
            RateLimitDecision {
                allowed: true,
                headers: RateLimitHeaders {
                    limit: self.max_requests,
                    remaining: self.max_requests - count - 1,
                    reset: reset_secs,
                    retry_after_ms: None,
                },
            }
        } else {
            // Earliest entry in window determines when space opens.
            let earliest = log.first().copied().unwrap_or(now_ms);
            let retry_after = (earliest + self.window_ms).saturating_sub(now_ms);
            RateLimitDecision {
                allowed: false,
                headers: RateLimitHeaders {
                    limit: self.max_requests,
                    remaining: 0,
                    reset: reset_secs,
                    retry_after_ms: Some(retry_after),
                },
            }
        }
    }

    /// Get request count in current window.
    pub fn current_count(&self, key: &str, now_ms: u64) -> u64 {
        let window_start = now_ms.saturating_sub(self.window_ms);
        self.logs
            .get(key)
            .map(|log| log.iter().filter(|&&ts| ts > window_start).count() as u64)
            .unwrap_or(0)
    }

    /// Clear all tracking data.
    pub fn clear(&mut self) {
        self.logs.clear();
    }

    /// Number of tracked keys.
    pub fn key_count(&self) -> usize {
        self.logs.len()
    }
}

// ── Fixed Window Counter ────────────────────────────────────────

/// Fixed window rate limiter.
///
/// Divides time into discrete windows and counts requests per window.
/// Simpler than sliding window, but can allow 2x burst at window boundaries.
#[derive(Debug, Clone)]
pub struct FixedWindowCounter {
    /// Maximum requests per window.
    max_requests: u64,
    /// Window size in milliseconds.
    window_ms: u64,
    /// Per-key state: (window_start_ms, count).
    windows: HashMap<String, (u64, u64)>,
}

impl FixedWindowCounter {
    /// Create a new fixed window limiter.
    pub fn new(max_requests: u64, window_ms: u64) -> Result<Self, RateLimiterError> {
        if max_requests == 0 {
            return Err(RateLimiterError::InvalidConfig(
                "max_requests must be > 0".to_string(),
            ));
        }
        if window_ms == 0 {
            return Err(RateLimiterError::InvalidConfig(
                "window_ms must be > 0".to_string(),
            ));
        }
        Ok(Self {
            max_requests,
            window_ms,
            windows: HashMap::new(),
        })
    }

    /// Compute the window start for a given timestamp.
    fn window_start(&self, now_ms: u64) -> u64 {
        (now_ms / self.window_ms) * self.window_ms
    }

    /// Check and record a request.
    pub fn check(&mut self, key: &str, now_ms: u64) -> RateLimitDecision {
        let ws = self.window_start(now_ms);
        let entry = self.windows.entry(key.to_string()).or_insert((ws, 0));

        // Reset if we moved to a new window.
        if entry.0 != ws {
            *entry = (ws, 0);
        }

        let reset_secs = (ws + self.window_ms) / 1000;

        if entry.1 < self.max_requests {
            entry.1 += 1;
            RateLimitDecision {
                allowed: true,
                headers: RateLimitHeaders {
                    limit: self.max_requests,
                    remaining: self.max_requests - entry.1,
                    reset: reset_secs,
                    retry_after_ms: None,
                },
            }
        } else {
            let retry_after = (ws + self.window_ms).saturating_sub(now_ms);
            RateLimitDecision {
                allowed: false,
                headers: RateLimitHeaders {
                    limit: self.max_requests,
                    remaining: 0,
                    reset: reset_secs,
                    retry_after_ms: Some(retry_after),
                },
            }
        }
    }

    /// Current count in the active window.
    pub fn current_count(&self, key: &str, now_ms: u64) -> u64 {
        let ws = self.window_start(now_ms);
        self.windows
            .get(key)
            .filter(|(start, _)| *start == ws)
            .map(|(_, count)| *count)
            .unwrap_or(0)
    }

    /// Remove tracking for a key.
    pub fn remove_key(&mut self, key: &str) {
        self.windows.remove(key);
    }
}

// ── Leaky Bucket ────────────────────────────────────────────────

/// Leaky bucket rate limiter.
///
/// Models a bucket that "leaks" at a constant rate. Requests fill the
/// bucket; if it overflows, the request is rejected.
#[derive(Debug, Clone)]
pub struct LeakyBucket {
    /// Maximum bucket capacity (queue depth).
    capacity: u64,
    /// Leak rate: requests drained per second.
    leak_rate: f64,
    /// Per-key state.
    buckets: HashMap<String, LeakyBucketState>,
}

#[derive(Debug, Clone)]
struct LeakyBucketState {
    /// Current water level.
    level: f64,
    /// Last update time in milliseconds.
    last_update_ms: u64,
}

impl LeakyBucket {
    /// Create a new leaky bucket limiter.
    pub fn new(capacity: u64, leak_rate: f64) -> Result<Self, RateLimiterError> {
        if capacity == 0 {
            return Err(RateLimiterError::InvalidConfig(
                "capacity must be > 0".to_string(),
            ));
        }
        if leak_rate <= 0.0 {
            return Err(RateLimiterError::InvalidConfig(
                "leak_rate must be > 0".to_string(),
            ));
        }
        Ok(Self {
            capacity,
            leak_rate,
            buckets: HashMap::new(),
        })
    }

    /// Check and record a request.
    pub fn check(&mut self, key: &str, now_ms: u64) -> RateLimitDecision {
        let state = self
            .buckets
            .entry(key.to_string())
            .or_insert(LeakyBucketState {
                level: 0.0,
                last_update_ms: now_ms,
            });

        // Drain based on elapsed time.
        let elapsed_ms = now_ms.saturating_sub(state.last_update_ms);
        let drained = (elapsed_ms as f64 / 1000.0) * self.leak_rate;
        state.level = (state.level - drained).max(0.0);
        state.last_update_ms = now_ms;

        let cap = self.capacity as f64;

        if state.level + 1.0 <= cap {
            state.level += 1.0;
            let remaining = (cap - state.level).max(0.0) as u64;
            RateLimitDecision {
                allowed: true,
                headers: RateLimitHeaders {
                    limit: self.capacity,
                    remaining,
                    reset: now_ms / 1000 + 1,
                    retry_after_ms: None,
                },
            }
        } else {
            // Time until 1 unit drains.
            let wait_ms = ((1.0 / self.leak_rate) * 1000.0).ceil() as u64;
            RateLimitDecision {
                allowed: false,
                headers: RateLimitHeaders {
                    limit: self.capacity,
                    remaining: 0,
                    reset: (now_ms + wait_ms) / 1000 + 1,
                    retry_after_ms: Some(wait_ms),
                },
            }
        }
    }

    /// Current level for a key.
    pub fn current_level(&self, key: &str) -> Option<f64> {
        self.buckets.get(key).map(|s| s.level)
    }

    /// Remove tracking for a key.
    pub fn remove_key(&mut self, key: &str) {
        self.buckets.remove(key);
    }

    /// Number of tracked keys.
    pub fn key_count(&self) -> usize {
        self.buckets.len()
    }
}

// ── Multi-Key Rate Limiter ──────────────────────────────────────

/// Strategy for the multi-key rate limiter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    TokenBucket,
    SlidingWindow,
    FixedWindow,
    LeakyBucket,
}

/// Configuration for the multi-key rate limiter.
#[derive(Debug, Clone)]
pub struct RateLimiterConfig {
    /// Strategy to use.
    pub strategy: Strategy,
    /// Maximum requests per window (or capacity for bucket strategies).
    pub max_requests: u64,
    /// Window/interval in milliseconds.
    pub window_ms: u64,
    /// Burst allowance (only for token bucket; defaults to max_requests).
    pub burst: Option<u64>,
}

impl RateLimiterConfig {
    /// Create a default config with sliding window strategy.
    pub fn new(max_requests: u64, window_ms: u64) -> Self {
        Self {
            strategy: Strategy::SlidingWindow,
            max_requests,
            window_ms,
            burst: None,
        }
    }

    /// Set the strategy.
    pub fn with_strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set burst allowance for token bucket.
    pub fn with_burst(mut self, burst: u64) -> Self {
        self.burst = Some(burst);
        self
    }
}

/// Multi-strategy per-key rate limiter facade.
pub enum MultiRateLimiter {
    TokenBucket(TokenBucket),
    SlidingWindow(SlidingWindowLog),
    FixedWindow(FixedWindowCounter),
    LeakyBucket(LeakyBucket),
}

impl MultiRateLimiter {
    /// Create from configuration.
    pub fn from_config(config: &RateLimiterConfig) -> Result<Self, RateLimiterError> {
        match config.strategy {
            Strategy::TokenBucket => {
                let capacity = config.burst.unwrap_or(config.max_requests) as f64;
                let refill_rate = config.max_requests as f64 / (config.window_ms as f64 / 1000.0);
                Ok(Self::TokenBucket(TokenBucket::new(capacity, refill_rate)?))
            }
            Strategy::SlidingWindow => Ok(Self::SlidingWindow(SlidingWindowLog::new(
                config.max_requests,
                config.window_ms,
            )?)),
            Strategy::FixedWindow => Ok(Self::FixedWindow(FixedWindowCounter::new(
                config.max_requests,
                config.window_ms,
            )?)),
            Strategy::LeakyBucket => {
                let leak_rate =
                    config.max_requests as f64 / (config.window_ms as f64 / 1000.0);
                Ok(Self::LeakyBucket(LeakyBucket::new(
                    config.max_requests,
                    leak_rate,
                )?))
            }
        }
    }

    /// Check a request for a key.
    pub fn check(&mut self, key: &str, now_ms: u64) -> RateLimitDecision {
        match self {
            Self::TokenBucket(tb) => tb.check(key, now_ms),
            Self::SlidingWindow(sw) => sw.check(key, now_ms),
            Self::FixedWindow(fw) => fw.check(key, now_ms),
            Self::LeakyBucket(lb) => lb.check(key, now_ms),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Token Bucket ──

    #[test]
    fn token_bucket_allows_within_capacity() {
        let mut tb = TokenBucket::new(5.0, 1.0).unwrap();
        for i in 0..5 {
            let d = tb.check("k", 1000 + i);
            assert!(d.allowed, "request {i} should be allowed");
        }
    }

    #[test]
    fn token_bucket_rejects_over_capacity() {
        let mut tb = TokenBucket::new(3.0, 1.0).unwrap();
        for _ in 0..3 {
            tb.check("k", 1000);
        }
        let d = tb.check("k", 1000);
        assert!(!d.allowed);
        assert!(d.headers.retry_after_ms.is_some());
    }

    #[test]
    fn token_bucket_refills_over_time() {
        let mut tb = TokenBucket::new(2.0, 2.0).unwrap();
        tb.check("k", 1000); // 1 token used
        tb.check("k", 1000); // 2 tokens used, bucket empty
        let d = tb.check("k", 1000);
        assert!(!d.allowed);
        // After 1 second, 2 tokens should refill.
        let d = tb.check("k", 2000);
        assert!(d.allowed);
    }

    #[test]
    fn token_bucket_per_key_isolation() {
        let mut tb = TokenBucket::new(1.0, 1.0).unwrap();
        let d1 = tb.check("a", 1000);
        let d2 = tb.check("b", 1000);
        assert!(d1.allowed);
        assert!(d2.allowed);
        let d3 = tb.check("a", 1000);
        assert!(!d3.allowed);
    }

    #[test]
    fn token_bucket_invalid_config() {
        assert!(TokenBucket::new(0.0, 1.0).is_err());
        assert!(TokenBucket::new(1.0, 0.0).is_err());
        assert!(TokenBucket::new(-1.0, 1.0).is_err());
    }

    #[test]
    fn token_bucket_try_consume_multiple() {
        let mut tb = TokenBucket::new(10.0, 1.0).unwrap();
        let d = tb.try_consume("k", 5.0, 1000);
        assert!(d.allowed);
        assert_eq!(d.headers.remaining, 5);
        let d = tb.try_consume("k", 6.0, 1000);
        assert!(!d.allowed);
    }

    // ── Sliding Window ──

    #[test]
    fn sliding_window_allows_within_limit() {
        let mut sw = SlidingWindowLog::new(3, 1000).unwrap();
        for i in 0..3 {
            let d = sw.check("k", 100 + i);
            assert!(d.allowed);
        }
    }

    #[test]
    fn sliding_window_rejects_over_limit() {
        let mut sw = SlidingWindowLog::new(2, 1000).unwrap();
        sw.check("k", 100);
        sw.check("k", 200);
        let d = sw.check("k", 300);
        assert!(!d.allowed);
    }

    #[test]
    fn sliding_window_resets_after_window() {
        let mut sw = SlidingWindowLog::new(1, 1000).unwrap();
        sw.check("k", 100);
        let d = sw.check("k", 200);
        assert!(!d.allowed);
        // After the window expires.
        let d = sw.check("k", 1200);
        assert!(d.allowed);
    }

    #[test]
    fn sliding_window_current_count() {
        let mut sw = SlidingWindowLog::new(10, 1000).unwrap();
        sw.check("k", 100);
        sw.check("k", 200);
        assert_eq!(sw.current_count("k", 500), 2);
        assert_eq!(sw.current_count("k", 1300), 0);
    }

    // ── Fixed Window ──

    #[test]
    fn fixed_window_allows_within_limit() {
        let mut fw = FixedWindowCounter::new(3, 1000).unwrap();
        for _ in 0..3 {
            let d = fw.check("k", 500);
            assert!(d.allowed);
        }
    }

    #[test]
    fn fixed_window_rejects_over_limit() {
        let mut fw = FixedWindowCounter::new(2, 1000).unwrap();
        fw.check("k", 500);
        fw.check("k", 600);
        let d = fw.check("k", 700);
        assert!(!d.allowed);
    }

    #[test]
    fn fixed_window_resets_on_new_window() {
        let mut fw = FixedWindowCounter::new(1, 1000).unwrap();
        fw.check("k", 500);
        let d = fw.check("k", 600);
        assert!(!d.allowed);
        // Next window starts at 1000.
        let d = fw.check("k", 1000);
        assert!(d.allowed);
    }

    #[test]
    fn fixed_window_current_count() {
        let mut fw = FixedWindowCounter::new(10, 1000).unwrap();
        fw.check("k", 100);
        fw.check("k", 200);
        assert_eq!(fw.current_count("k", 500), 2);
        // New window.
        assert_eq!(fw.current_count("k", 1500), 0);
    }

    // ── Leaky Bucket ──

    #[test]
    fn leaky_bucket_allows_within_capacity() {
        let mut lb = LeakyBucket::new(5, 1.0).unwrap();
        for _ in 0..5 {
            let d = lb.check("k", 1000);
            assert!(d.allowed);
        }
    }

    #[test]
    fn leaky_bucket_rejects_overflow() {
        let mut lb = LeakyBucket::new(2, 1.0).unwrap();
        lb.check("k", 1000);
        lb.check("k", 1000);
        let d = lb.check("k", 1000);
        assert!(!d.allowed);
    }

    #[test]
    fn leaky_bucket_drains_over_time() {
        let mut lb = LeakyBucket::new(1, 1.0).unwrap();
        lb.check("k", 1000);
        let d = lb.check("k", 1000);
        assert!(!d.allowed);
        // After 1 second, 1 unit drained.
        let d = lb.check("k", 2000);
        assert!(d.allowed);
    }

    #[test]
    fn leaky_bucket_invalid_config() {
        assert!(LeakyBucket::new(0, 1.0).is_err());
        assert!(LeakyBucket::new(1, 0.0).is_err());
    }

    // ── Headers ──

    #[test]
    fn rate_limit_headers_to_pairs() {
        let h = RateLimitHeaders {
            limit: 100,
            remaining: 99,
            reset: 1700000000,
            retry_after_ms: None,
        };
        let pairs = h.to_header_pairs();
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].0, "X-RateLimit-Limit");
        assert_eq!(pairs[0].1, "100");
    }

    #[test]
    fn rate_limit_headers_with_retry_after() {
        let h = RateLimitHeaders {
            limit: 10,
            remaining: 0,
            reset: 1700000000,
            retry_after_ms: Some(1500),
        };
        let pairs = h.to_header_pairs();
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[3].0, "Retry-After");
        assert_eq!(pairs[3].1, "2"); // ceil(1500/1000)
    }

    // ── Multi Rate Limiter ──

    #[test]
    fn multi_limiter_sliding_window() {
        let config = RateLimiterConfig::new(2, 1000);
        let mut limiter = MultiRateLimiter::from_config(&config).unwrap();
        assert!(limiter.check("k", 100).allowed);
        assert!(limiter.check("k", 200).allowed);
        assert!(!limiter.check("k", 300).allowed);
    }

    #[test]
    fn multi_limiter_token_bucket() {
        let config = RateLimiterConfig::new(2, 1000).with_strategy(Strategy::TokenBucket);
        let mut limiter = MultiRateLimiter::from_config(&config).unwrap();
        assert!(limiter.check("k", 100).allowed);
        assert!(limiter.check("k", 200).allowed);
        assert!(!limiter.check("k", 300).allowed);
    }

    #[test]
    fn multi_limiter_fixed_window() {
        let config = RateLimiterConfig::new(1, 1000).with_strategy(Strategy::FixedWindow);
        let mut limiter = MultiRateLimiter::from_config(&config).unwrap();
        assert!(limiter.check("k", 100).allowed);
        assert!(!limiter.check("k", 200).allowed);
    }

    #[test]
    fn multi_limiter_leaky_bucket() {
        let config = RateLimiterConfig::new(1, 1000).with_strategy(Strategy::LeakyBucket);
        let mut limiter = MultiRateLimiter::from_config(&config).unwrap();
        assert!(limiter.check("k", 100).allowed);
        assert!(!limiter.check("k", 200).allowed);
    }

    #[test]
    fn multi_limiter_with_burst() {
        let config = RateLimiterConfig::new(2, 1000)
            .with_strategy(Strategy::TokenBucket)
            .with_burst(5);
        let mut limiter = MultiRateLimiter::from_config(&config).unwrap();
        // Burst capacity is 5.
        for _ in 0..5 {
            assert!(limiter.check("k", 100).allowed);
        }
        assert!(!limiter.check("k", 100).allowed);
    }

    #[test]
    fn error_display() {
        let e = RateLimiterError::LimitExceeded {
            retry_after_ms: 500,
        };
        assert!(e.to_string().contains("500"));
        let e2 = RateLimiterError::InvalidConfig("bad".to_string());
        assert!(e2.to_string().contains("bad"));
    }

    #[test]
    fn token_bucket_remove_key() {
        let mut tb = TokenBucket::new(5.0, 1.0).unwrap();
        tb.check("a", 1000);
        assert_eq!(tb.key_count(), 1);
        tb.remove_key("a");
        assert_eq!(tb.key_count(), 0);
    }

    #[test]
    fn sliding_window_clear() {
        let mut sw = SlidingWindowLog::new(10, 1000).unwrap();
        sw.check("a", 100);
        sw.check("b", 200);
        assert_eq!(sw.key_count(), 2);
        sw.clear();
        assert_eq!(sw.key_count(), 0);
    }

    #[test]
    fn remaining_decreases() {
        let mut fw = FixedWindowCounter::new(5, 1000).unwrap();
        let d1 = fw.check("k", 100);
        assert_eq!(d1.headers.remaining, 4);
        let d2 = fw.check("k", 200);
        assert_eq!(d2.headers.remaining, 3);
    }
}
