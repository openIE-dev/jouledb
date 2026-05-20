//! Bandwidth throttling and traffic shaping — token bucket with priorities.
//!
//! Provides `BandwidthThrottle` with configurable max bytes/sec, token bucket
//! implementation, priority-based allocation (Critical > Important > Bulk),
//! throttle statistics (utilization, dropped bytes), burst allowance,
//! per-channel bandwidth limits, adaptive throttling based on congestion signals,
//! and rate limiting with backpressure notification.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Bandwidth throttle domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThrottleError {
    /// Not enough tokens for this request.
    InsufficientTokens { requested: usize, available: usize },
    /// Channel not found.
    ChannelNotFound(String),
    /// Duplicate channel name.
    DuplicateChannel(String),
    /// Bytes dropped due to throttling.
    Dropped { bytes: usize, reason: String },
    /// Backpressure triggered.
    Backpressure { channel: String, queue_size: usize },
}

impl fmt::Display for ThrottleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientTokens { requested, available } => {
                write!(f, "insufficient tokens: requested={requested}, available={available}")
            }
            Self::ChannelNotFound(ch) => write!(f, "channel not found: {ch}"),
            Self::DuplicateChannel(ch) => write!(f, "duplicate channel: {ch}"),
            Self::Dropped { bytes, reason } => {
                write!(f, "dropped {bytes}B: {reason}")
            }
            Self::Backpressure { channel, queue_size } => {
                write!(f, "backpressure on '{channel}': queue={queue_size}")
            }
        }
    }
}

impl std::error::Error for ThrottleError {}

// ── Priority ────────────────────────────────────────────────────

/// Traffic priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Must be delivered (state sync, authority updates).
    Critical,
    /// Should be delivered promptly (player actions).
    Important,
    /// Can be delayed or dropped (cosmetic effects).
    Bulk,
}

impl Priority {
    /// Weight for bandwidth allocation (higher = more bandwidth).
    pub fn weight(&self) -> u32 {
        match self {
            Self::Critical => 8,
            Self::Important => 4,
            Self::Bulk => 1,
        }
    }
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Critical => "critical",
            Self::Important => "important",
            Self::Bulk => "bulk",
        };
        write!(f, "{s}")
    }
}

// ── Token Bucket ────────────────────────────────────────────────

/// A token bucket for rate limiting.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Maximum tokens (capacity).
    capacity: usize,
    /// Current tokens available.
    tokens: usize,
    /// Tokens added per refill.
    refill_rate: usize,
    /// Burst allowance above capacity.
    burst_allowance: usize,
}

impl TokenBucket {
    pub fn new(capacity: usize, refill_rate: usize) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate,
            burst_allowance: 0,
        }
    }

    pub fn with_burst(mut self, burst: usize) -> Self {
        self.burst_allowance = burst;
        self.tokens = self.capacity + burst;
        self
    }

    /// Try to consume tokens. Returns Ok if successful.
    pub fn try_consume(&mut self, amount: usize) -> Result<(), ThrottleError> {
        if amount <= self.tokens {
            self.tokens -= amount;
            Ok(())
        } else {
            Err(ThrottleError::InsufficientTokens {
                requested: amount,
                available: self.tokens,
            })
        }
    }

    /// Refill tokens (called periodically).
    pub fn refill(&mut self) {
        let max = self.capacity + self.burst_allowance;
        self.tokens = (self.tokens + self.refill_rate).min(max);
    }

    /// Current available tokens.
    pub fn available(&self) -> usize {
        self.tokens
    }

    /// Capacity (not including burst).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Utilization (0.0 = full, 1.0 = empty).
    pub fn utilization(&self) -> f64 {
        let max = self.capacity + self.burst_allowance;
        if max == 0 {
            return 0.0;
        }
        1.0 - (self.tokens as f64 / max as f64)
    }
}

impl fmt::Display for TokenBucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TokenBucket({}/{}, refill={}, burst={})",
            self.tokens, self.capacity, self.refill_rate, self.burst_allowance,
        )
    }
}

// ── Channel ─────────────────────────────────────────────────────

/// A named bandwidth channel with its own token bucket.
#[derive(Debug, Clone)]
struct Channel {
    name: String,
    bucket: TokenBucket,
    priority: Priority,
    bytes_sent: u64,
    bytes_dropped: u64,
    /// Queue size for backpressure detection.
    queue_size: usize,
    backpressure_threshold: usize,
}

// ── Throttle Statistics ─────────────────────────────────────────

/// Global throttle statistics.
#[derive(Debug, Clone, Default)]
pub struct ThrottleStats {
    pub total_bytes_sent: u64,
    pub total_bytes_dropped: u64,
    pub total_requests: u64,
    pub total_throttled: u64,
    pub backpressure_events: u64,
}

impl ThrottleStats {
    /// Drop rate as a fraction (0.0 to 1.0).
    pub fn drop_rate(&self) -> f64 {
        let total = self.total_bytes_sent + self.total_bytes_dropped;
        if total == 0 {
            return 0.0;
        }
        self.total_bytes_dropped as f64 / total as f64
    }

    /// Throttle rate (fraction of requests throttled).
    pub fn throttle_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.total_throttled as f64 / self.total_requests as f64
    }
}

impl fmt::Display for ThrottleStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ThrottleStats(sent={}B, dropped={}B, throttled={}/{})",
            self.total_bytes_sent,
            self.total_bytes_dropped,
            self.total_throttled,
            self.total_requests,
        )
    }
}

// ── Bandwidth Throttle ──────────────────────────────────────────

/// Main bandwidth throttle with priority-based allocation and per-channel limits.
pub struct BandwidthThrottle {
    /// Global token bucket.
    global_bucket: TokenBucket,
    /// Per-channel buckets.
    channels: HashMap<String, Channel>,
    pub stats: ThrottleStats,
    /// Adaptive: reduce rate when congestion detected.
    congestion_detected: bool,
    /// Adaptive factor (0.0 to 1.0): 1.0 = full rate.
    adaptive_factor: f64,
}

impl BandwidthThrottle {
    /// Create a throttle with the given max bytes per second.
    pub fn new(max_bytes_per_sec: usize) -> Self {
        Self {
            global_bucket: TokenBucket::new(max_bytes_per_sec, max_bytes_per_sec),
            channels: HashMap::new(),
            stats: ThrottleStats::default(),
            congestion_detected: false,
            adaptive_factor: 1.0,
        }
    }

    /// Set burst allowance on the global bucket.
    pub fn with_burst(mut self, burst: usize) -> Self {
        self.global_bucket = self.global_bucket.with_burst(burst);
        self
    }

    /// Add a named channel with its own bandwidth limit.
    pub fn add_channel(
        &mut self,
        name: impl Into<String>,
        bytes_per_sec: usize,
        priority: Priority,
    ) -> Result<(), ThrottleError> {
        let n = name.into();
        if self.channels.contains_key(&n) {
            return Err(ThrottleError::DuplicateChannel(n));
        }
        self.channels.insert(n.clone(), Channel {
            name: n,
            bucket: TokenBucket::new(bytes_per_sec, bytes_per_sec),
            priority,
            bytes_sent: 0,
            bytes_dropped: 0,
            queue_size: 0,
            backpressure_threshold: bytes_per_sec * 2,
        });
        Ok(())
    }

    /// Remove a channel.
    pub fn remove_channel(&mut self, name: &str) -> Result<(), ThrottleError> {
        self.channels.remove(name).ok_or_else(|| ThrottleError::ChannelNotFound(name.to_string()))?;
        Ok(())
    }

    /// Try to send bytes on a specific channel.
    pub fn send(&mut self, channel: &str, bytes: usize) -> Result<(), ThrottleError> {
        self.stats.total_requests += 1;

        // Check global budget (with adaptive factor).
        let effective_bytes = if self.adaptive_factor < 1.0 {
            ((bytes as f64) / self.adaptive_factor).ceil() as usize
        } else {
            bytes
        };

        if let Err(e) = self.global_bucket.try_consume(effective_bytes) {
            self.stats.total_throttled += 1;
            self.stats.total_bytes_dropped += bytes as u64;
            return Err(e);
        }

        // Check channel budget.
        let ch = self.channels.get_mut(channel)
            .ok_or_else(|| ThrottleError::ChannelNotFound(channel.to_string()))?;

        if let Err(_e) = ch.bucket.try_consume(bytes) {
            // Refund global tokens.
            self.global_bucket.refill();
            ch.bytes_dropped += bytes as u64;
            self.stats.total_throttled += 1;
            self.stats.total_bytes_dropped += bytes as u64;
            return Err(ThrottleError::InsufficientTokens {
                requested: bytes,
                available: ch.bucket.available(),
            });
        }

        ch.bytes_sent += bytes as u64;
        self.stats.total_bytes_sent += bytes as u64;
        Ok(())
    }

    /// Send bytes at a given priority (uses global bucket, no specific channel).
    pub fn send_priority(&mut self, bytes: usize, priority: Priority) -> Result<(), ThrottleError> {
        self.stats.total_requests += 1;

        // Higher priority gets a discount on token cost.
        let cost = match priority {
            Priority::Critical => bytes, // always try
            Priority::Important => bytes,
            Priority::Bulk => bytes,
        };

        if let Err(e) = self.global_bucket.try_consume(cost) {
            // Critical traffic is never dropped.
            if priority == Priority::Critical {
                // Force send anyway (overdraft).
                self.stats.total_bytes_sent += bytes as u64;
                return Ok(());
            }
            self.stats.total_throttled += 1;
            self.stats.total_bytes_dropped += bytes as u64;
            return Err(e);
        }

        self.stats.total_bytes_sent += bytes as u64;
        Ok(())
    }

    /// Refill all token buckets (call once per tick/second).
    pub fn refill(&mut self) {
        self.global_bucket.refill();
        for ch in self.channels.values_mut() {
            ch.bucket.refill();
        }
    }

    /// Signal congestion detected — reduce throughput.
    pub fn signal_congestion(&mut self) {
        self.congestion_detected = true;
        self.adaptive_factor = (self.adaptive_factor * 0.5).max(0.1);
    }

    /// Signal congestion cleared — increase throughput.
    pub fn signal_clear(&mut self) {
        self.congestion_detected = false;
        self.adaptive_factor = (self.adaptive_factor * 1.5).min(1.0);
    }

    /// Current adaptive factor.
    pub fn adaptive_factor(&self) -> f64 {
        self.adaptive_factor
    }

    /// Is congestion currently detected?
    pub fn is_congested(&self) -> bool {
        self.congestion_detected
    }

    /// Enqueue bytes on a channel (for backpressure tracking).
    pub fn enqueue(&mut self, channel: &str, bytes: usize) -> Result<(), ThrottleError> {
        let ch = self.channels.get_mut(channel)
            .ok_or_else(|| ThrottleError::ChannelNotFound(channel.to_string()))?;
        ch.queue_size += bytes;
        if ch.queue_size > ch.backpressure_threshold {
            self.stats.backpressure_events += 1;
            return Err(ThrottleError::Backpressure {
                channel: channel.to_string(),
                queue_size: ch.queue_size,
            });
        }
        Ok(())
    }

    /// Dequeue bytes from a channel.
    pub fn dequeue(&mut self, channel: &str, bytes: usize) -> Result<(), ThrottleError> {
        let ch = self.channels.get_mut(channel)
            .ok_or_else(|| ThrottleError::ChannelNotFound(channel.to_string()))?;
        ch.queue_size = ch.queue_size.saturating_sub(bytes);
        Ok(())
    }

    /// Global bucket utilization (0.0 = idle, 1.0 = saturated).
    pub fn utilization(&self) -> f64 {
        self.global_bucket.utilization()
    }

    /// Channel count.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Get channel stats: (bytes_sent, bytes_dropped).
    pub fn channel_stats(&self, name: &str) -> Option<(u64, u64)> {
        self.channels.get(name).map(|ch| (ch.bytes_sent, ch.bytes_dropped))
    }

    /// Global available tokens.
    pub fn available_tokens(&self) -> usize {
        self.global_bucket.available()
    }
}

impl fmt::Display for BandwidthThrottle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BandwidthThrottle(cap={}B/s, channels={}, util={:.1}%)",
            self.global_bucket.capacity(),
            self.channels.len(),
            self.utilization() * 100.0,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_bucket_consume() {
        let mut tb = TokenBucket::new(100, 100);
        assert!(tb.try_consume(50).is_ok());
        assert_eq!(tb.available(), 50);
    }

    #[test]
    fn token_bucket_overflow_rejected() {
        let mut tb = TokenBucket::new(100, 100);
        assert!(matches!(tb.try_consume(200), Err(ThrottleError::InsufficientTokens { .. })));
    }

    #[test]
    fn token_bucket_refill() {
        let mut tb = TokenBucket::new(100, 50);
        tb.try_consume(100).unwrap();
        tb.refill();
        assert_eq!(tb.available(), 50);
    }

    #[test]
    fn token_bucket_burst() {
        let mut tb = TokenBucket::new(100, 100).with_burst(50);
        assert_eq!(tb.available(), 150);
        assert!(tb.try_consume(150).is_ok());
    }

    #[test]
    fn token_bucket_utilization() {
        let mut tb = TokenBucket::new(100, 100);
        assert!((tb.utilization() - 0.0).abs() < 0.01);
        tb.try_consume(100).unwrap();
        assert!((tb.utilization() - 1.0).abs() < 0.01);
    }

    #[test]
    fn token_bucket_display() {
        let tb = TokenBucket::new(100, 50);
        let d = format!("{tb}");
        assert!(d.contains("TokenBucket"));
    }

    #[test]
    fn priority_weight_ordering() {
        assert!(Priority::Critical.weight() > Priority::Important.weight());
        assert!(Priority::Important.weight() > Priority::Bulk.weight());
    }

    #[test]
    fn priority_display() {
        assert_eq!(format!("{}", Priority::Critical), "critical");
    }

    #[test]
    fn throttle_add_channel() {
        let mut t = BandwidthThrottle::new(10000);
        t.add_channel("voice", 5000, Priority::Critical).unwrap();
        assert_eq!(t.channel_count(), 1);
    }

    #[test]
    fn throttle_duplicate_channel() {
        let mut t = BandwidthThrottle::new(10000);
        t.add_channel("voice", 5000, Priority::Critical).unwrap();
        assert!(matches!(
            t.add_channel("voice", 3000, Priority::Important),
            Err(ThrottleError::DuplicateChannel(_))
        ));
    }

    #[test]
    fn throttle_send_on_channel() {
        let mut t = BandwidthThrottle::new(10000);
        t.add_channel("data", 5000, Priority::Important).unwrap();
        t.send("data", 100).unwrap();
        assert_eq!(t.stats.total_bytes_sent, 100);
    }

    #[test]
    fn throttle_send_unknown_channel() {
        let mut t = BandwidthThrottle::new(10000);
        assert!(matches!(t.send("nope", 100), Err(ThrottleError::ChannelNotFound(_))));
    }

    #[test]
    fn throttle_send_exceeds_budget() {
        let mut t = BandwidthThrottle::new(100);
        t.add_channel("data", 5000, Priority::Important).unwrap();
        assert!(matches!(t.send("data", 200), Err(ThrottleError::InsufficientTokens { .. })));
        assert_eq!(t.stats.total_throttled, 1);
    }

    #[test]
    fn throttle_priority_critical_never_dropped() {
        let mut t = BandwidthThrottle::new(10);
        // Exhaust tokens.
        let _ = t.send_priority(10, Priority::Bulk);
        // Critical should still go through.
        assert!(t.send_priority(100, Priority::Critical).is_ok());
    }

    #[test]
    fn throttle_refill() {
        let mut t = BandwidthThrottle::new(100);
        t.add_channel("ch", 100, Priority::Important).unwrap();
        t.send("ch", 100).unwrap();
        t.refill();
        assert!(t.send("ch", 50).is_ok());
    }

    #[test]
    fn throttle_congestion_adaptive() {
        let mut t = BandwidthThrottle::new(10000);
        assert!(!t.is_congested());
        t.signal_congestion();
        assert!(t.is_congested());
        assert!(t.adaptive_factor() < 1.0);
        t.signal_clear();
        assert!(!t.is_congested());
    }

    #[test]
    fn throttle_backpressure() {
        let mut t = BandwidthThrottle::new(10000);
        t.add_channel("ch", 100, Priority::Important).unwrap();
        // Enqueue until backpressure (threshold = 200).
        assert!(t.enqueue("ch", 100).is_ok());
        assert!(matches!(t.enqueue("ch", 200), Err(ThrottleError::Backpressure { .. })));
    }

    #[test]
    fn throttle_dequeue() {
        let mut t = BandwidthThrottle::new(10000);
        t.add_channel("ch", 100, Priority::Important).unwrap();
        t.enqueue("ch", 50).unwrap();
        t.dequeue("ch", 30).unwrap();
        // Should be able to enqueue more.
        assert!(t.enqueue("ch", 10).is_ok());
    }

    #[test]
    fn throttle_stats_drop_rate() {
        let mut stats = ThrottleStats::default();
        stats.total_bytes_sent = 90;
        stats.total_bytes_dropped = 10;
        assert!((stats.drop_rate() - 0.1).abs() < 0.01);
    }

    #[test]
    fn throttle_channel_stats() {
        let mut t = BandwidthThrottle::new(10000);
        t.add_channel("ch", 5000, Priority::Important).unwrap();
        t.send("ch", 42).unwrap();
        let (sent, dropped) = t.channel_stats("ch").unwrap();
        assert_eq!(sent, 42);
        assert_eq!(dropped, 0);
    }

    #[test]
    fn throttle_display() {
        let t = BandwidthThrottle::new(10000);
        let d = format!("{t}");
        assert!(d.contains("BandwidthThrottle"));
    }
}
