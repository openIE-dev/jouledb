use crate::compression::CompressionId;
use crate::encoding::EncodingId;
use crate::frame::{FrameType, HeaderFormat, RateLimitPayload};

/// Tracks rate limit state from a server's RateLimit advisory frame.
#[derive(Debug, Clone)]
pub struct RateLimitInfo {
    /// Queries remaining in the current window.
    pub queries_remaining: u64,
    /// Window duration in seconds.
    pub window_seconds: u32,
    /// If set, client should wait this many ms before next query.
    pub retry_after_ms: Option<u64>,
    /// If set, remaining energy budget for this window (µWh).
    pub energy_budget_uwh: Option<u64>,
}

/// Energy reporting granularity — how often the protocol includes
/// energy cost in frame headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EnergyReporting {
    /// Every frame carries its cumulative energy cost.
    PerFrame = 0,
    /// Energy reported once per query completion (Done frame).
    PerQuery = 1,
    /// Energy reported once per session close.
    PerSession = 2,
}

impl EnergyReporting {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::PerFrame),
            1 => Some(Self::PerQuery),
            2 => Some(Self::PerSession),
            _ => None,
        }
    }
}

/// Live connection profile — the brain of the adaptive protocol.
///
/// Observes per-frame statistics and derives optimal encoding, compression,
/// batching, and header format decisions. Updated continuously. All decisions
/// use integer math (no floats in decision paths).
///
/// # Adaptation rules
///
/// 1. **Compression**: Zstd when `avg_payload_bytes > 256`. Lz4 when
///    `avg_rtt_us < 5000` (local net — prefer speed). None when
///    `avg_payload_bytes < 64`.
/// 2. **Headers**: Compact for Heartbeat/Cancel. Extended for
///    Receipt/EnergyGradient. Standard otherwise.
/// 3. **Batching**: Batch results when `queries_served > 2` and
///    results per query > 3. Batch size = min(optimal_batch_size, total).
/// 4. **Energy reporting**: Starts PerFrame, downgrades to PerQuery
///    after 50 frames to reduce overhead.
pub struct ConnectionProfile {
    // ── Measured (updated per-frame) ─────────────────────────────
    /// Average round-trip time in microseconds (EWMA).
    pub avg_rtt_us: u64,
    /// Average payload size in bytes (EWMA).
    pub avg_payload_bytes: u64,
    /// Cache hit rate as `[0, 10000]` (basis points).
    pub cache_hit_rate: u16,
    /// Total queries completed on this connection.
    pub queries_served: u64,
    /// Cumulative energy consumed in µWh.
    pub cumulative_energy_uwh: u64,
    /// Total frames exchanged.
    pub frames_exchanged: u64,

    // ── Derived (recalculated after each observation) ────────────
    /// Optimal payload encoding.
    pub optimal_encoding: EncodingId,
    /// Optimal compression strategy.
    pub optimal_compression: CompressionId,
    /// Optimal number of results to batch per Batch frame.
    pub optimal_batch_size: u16,
    /// Default header format for data frames.
    pub header_format: HeaderFormat,
    /// Current energy reporting granularity.
    pub energy_reporting: EnergyReporting,

    /// Recommended query interval in ms, derived from energy gradient.
    /// Baseline 100ms; increases when costs are rising.
    pub recommended_query_interval_ms: u64,
    /// Energy threshold (µWh) below which a query is considered a cache hit.
    pub cache_hit_threshold_uwh: u64,

    /// Current rate limit info from the server, if any.
    pub rate_limit: Option<RateLimitInfo>,

    // ── Internal: energy gradient (rolling window) ───────────────
    energy_window: Vec<u64>,
    energy_window_capacity: usize,
    total_results_this_session: u64,
}

impl ConnectionProfile {
    /// Create a fresh profile with conservative defaults.
    ///
    /// Starts with: CBOR encoding, no compression, no batching,
    /// standard headers, per-frame energy reporting.
    pub fn new() -> Self {
        Self {
            avg_rtt_us: 0,
            avg_payload_bytes: 0,
            cache_hit_rate: 0,
            queries_served: 0,
            cumulative_energy_uwh: 0,
            frames_exchanged: 0,

            optimal_encoding: EncodingId::Cbor,
            optimal_compression: CompressionId::None,
            optimal_batch_size: 1,
            header_format: HeaderFormat::Standard,
            energy_reporting: EnergyReporting::PerFrame,

            recommended_query_interval_ms: 100,
            cache_hit_threshold_uwh: 50,
            rate_limit: None,

            energy_window: Vec::with_capacity(16),
            energy_window_capacity: 16,
            total_results_this_session: 0,
        }
    }

    /// Observe a single frame's payload size and energy cost.
    /// Recalculates derived decisions.
    pub fn observe_frame(&mut self, payload_bytes: u64, energy_uwh: u64, rtt_us: Option<u64>) {
        self.frames_exchanged += 1;
        self.cumulative_energy_uwh += energy_uwh;

        // EWMA for payload size (α = 1/8 in integer math: new = old*7/8 + sample/8)
        if self.frames_exchanged == 1 {
            self.avg_payload_bytes = payload_bytes;
        } else {
            self.avg_payload_bytes = (self.avg_payload_bytes * 7 + payload_bytes) / 8;
        }

        // EWMA for RTT
        if let Some(rtt) = rtt_us {
            if self.avg_rtt_us == 0 {
                self.avg_rtt_us = rtt;
            } else {
                self.avg_rtt_us = (self.avg_rtt_us * 7 + rtt) / 8;
            }
        }

        self.recalculate();
    }

    /// Observe a completed query's total energy and cache hit status.
    pub fn observe_query(&mut self, total_energy_uwh: u64, cache_hit: bool, result_count: u64) {
        self.queries_served += 1;
        self.total_results_this_session += result_count;

        // Push into energy window (ring buffer)
        if self.energy_window.len() < self.energy_window_capacity {
            self.energy_window.push(total_energy_uwh);
        } else {
            let idx = (self.queries_served as usize - 1) % self.energy_window_capacity;
            self.energy_window[idx] = total_energy_uwh;
        }

        // Update cache hit rate EWMA (in basis points, α = 1/4)
        let hit_bp: u16 = if cache_hit { 10000 } else { 0 };
        if self.queries_served == 1 {
            self.cache_hit_rate = hit_bp;
        } else {
            self.cache_hit_rate = ((self.cache_hit_rate as u32 * 3 + hit_bp as u32) / 4) as u16;
        }

        self.recalculate();
    }

    /// Energy gradient: µWh/query slope over the rolling window.
    /// Negative = getting cheaper. Returns 0 if window has < 2 entries.
    pub fn energy_gradient(&self) -> i64 {
        let n = self.energy_window.len();
        if n < 2 {
            return 0;
        }

        // Simple linear regression slope using integer math.
        // slope ≈ (last_half_avg - first_half_avg) / (n/2)
        let half = n / 2;
        let first_half_sum: u64 = self.energy_window[..half].iter().sum();
        let second_half_sum: u64 = self.energy_window[half..].iter().sum();

        let first_avg = first_half_sum / half as u64;
        let second_avg = second_half_sum / (n - half) as u64;

        second_avg as i64 - first_avg as i64
    }

    /// Number of entries in the energy window.
    pub fn energy_window_len(&self) -> usize {
        self.energy_window.len()
    }

    /// Whether the client should throttle queries due to rising energy costs.
    /// True when the energy gradient exceeds 50 µWh/query (costs spiking).
    pub fn should_throttle(&self) -> bool {
        self.energy_gradient() > 50
    }

    /// Update rate limit state from a server RateLimit advisory frame.
    pub fn update_rate_limit(&mut self, payload: &RateLimitPayload) {
        self.rate_limit = Some(RateLimitInfo {
            queries_remaining: payload.queries_remaining,
            window_seconds: payload.window_seconds,
            retry_after_ms: payload.retry_after_ms,
            energy_budget_uwh: payload.energy_budget_uwh,
        });
    }

    /// Whether the client is rate-limited: retry_after_ms is set or queries_remaining == 0.
    pub fn is_rate_limited(&self) -> bool {
        self.rate_limit
            .as_ref()
            .is_some_and(|rl| rl.retry_after_ms.is_some() || rl.queries_remaining == 0)
    }

    /// Choose the header format for a given frame type.
    pub fn header_for(&self, frame_type: FrameType) -> HeaderFormat {
        match frame_type {
            // Control frames: always compact (8 bytes vs 21)
            FrameType::Heartbeat | FrameType::Cancel => HeaderFormat::Compact,

            // Energy-rich frames: extended when we have breakdown data
            FrameType::Receipt | FrameType::EnergyGradient | FrameType::Done => {
                HeaderFormat::Extended
            }

            // Everything else: standard
            _ => HeaderFormat::Standard,
        }
    }

    /// Should we compress a payload of the given size?
    pub fn should_compress(&self, payload_bytes: usize) -> bool {
        match self.optimal_compression {
            CompressionId::None => false,
            CompressionId::Zstd => payload_bytes > 256,
            CompressionId::Lz4 => payload_bytes > 128,
        }
    }

    /// Optimal batch size for a set of results.
    pub fn batch_size_for_results(&self, total_results: usize) -> usize {
        if self.optimal_batch_size <= 1 || total_results <= 1 {
            return 1;
        }
        total_results.min(self.optimal_batch_size as usize)
    }

    /// Recalculate all derived decisions from measured values.
    fn recalculate(&mut self) {
        // ── Compression decision ─────────────────────────────────
        self.optimal_compression = if self.avg_payload_bytes < 64 {
            CompressionId::None
        } else if self.avg_rtt_us > 0 && self.avg_rtt_us < 5000 {
            // Local network: prefer speed
            CompressionId::Lz4
        } else if self.avg_payload_bytes > 256 {
            CompressionId::Zstd
        } else {
            CompressionId::None
        };

        // ── Batching decision ────────────────────────────────────
        let avg_results_per_query = if self.queries_served > 0 {
            self.total_results_this_session / self.queries_served
        } else {
            0
        };

        self.optimal_batch_size = if self.queries_served > 2 && avg_results_per_query > 3 {
            // Batch up to 16 results, saves 76% header overhead per batched result
            (avg_results_per_query as u16).clamp(2, 16)
        } else {
            1
        };

        // ── Energy reporting downgrade ───────────────────────────
        self.energy_reporting = if self.frames_exchanged > 50 {
            EnergyReporting::PerQuery
        } else {
            EnergyReporting::PerFrame
        };

        // ── Gradient-based query interval ─────────────────────────
        let gradient = self.energy_gradient();
        self.recommended_query_interval_ms = if gradient > 0 {
            // Costs rising: back off proportionally (100ms baseline + 10ms per µWh/query)
            100 + (gradient as u64).saturating_mul(10)
        } else {
            100 // Baseline: costs stable or falling
        };
    }
}

impl Default for ConnectionProfile {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_profile_conservative_defaults() {
        let p = ConnectionProfile::new();
        assert_eq!(p.optimal_encoding, EncodingId::Cbor);
        assert_eq!(p.optimal_compression, CompressionId::None);
        assert_eq!(p.optimal_batch_size, 1);
        assert_eq!(p.energy_reporting, EnergyReporting::PerFrame);
        assert_eq!(p.header_format, HeaderFormat::Standard);
    }

    #[test]
    fn compression_switches_to_zstd_for_large_payloads() {
        let mut p = ConnectionProfile::new();

        // Simulate 10 frames with avg_payload > 256
        for i in 0..10 {
            p.observe_frame(512, 100, Some(20_000));
            // After enough frames, avg_payload should be high
            if i >= 2 {
                assert!(
                    p.avg_payload_bytes > 256,
                    "avg_payload_bytes should be > 256 after {} frames, got {}",
                    i + 1,
                    p.avg_payload_bytes
                );
            }
        }

        assert_eq!(p.optimal_compression, CompressionId::Zstd);
    }

    #[test]
    fn compression_switches_to_lz4_for_low_rtt() {
        let mut p = ConnectionProfile::new();

        // Simulate frames with large payload but low RTT (local net)
        for _ in 0..10 {
            p.observe_frame(512, 100, Some(2000));
        }

        assert_eq!(p.optimal_compression, CompressionId::Lz4);
    }

    #[test]
    fn energy_reporting_downgrades_after_50_frames() {
        let mut p = ConnectionProfile::new();
        assert_eq!(p.energy_reporting, EnergyReporting::PerFrame);

        for _ in 0..51 {
            p.observe_frame(100, 10, None);
        }

        assert_eq!(p.energy_reporting, EnergyReporting::PerQuery);
    }

    #[test]
    fn energy_gradient_ascending() {
        let mut p = ConnectionProfile::new();

        // Query costs increasing: 100, 200, 300, 400
        for cost in [100, 200, 300, 400] {
            p.observe_query(cost, false, 5);
        }

        let gradient = p.energy_gradient();
        assert!(
            gradient > 0,
            "gradient should be positive (getting more expensive), got {gradient}"
        );
    }

    #[test]
    fn energy_gradient_descending() {
        let mut p = ConnectionProfile::new();

        // Query costs decreasing: 400, 300, 200, 100
        for cost in [400, 300, 200, 100] {
            p.observe_query(cost, false, 5);
        }

        let gradient = p.energy_gradient();
        assert!(
            gradient < 0,
            "gradient should be negative (getting cheaper), got {gradient}"
        );
    }

    #[test]
    fn header_for_heartbeat_returns_compact() {
        let p = ConnectionProfile::new();
        assert_eq!(p.header_for(FrameType::Heartbeat), HeaderFormat::Compact);
        assert_eq!(p.header_for(FrameType::Cancel), HeaderFormat::Compact);
    }

    #[test]
    fn header_for_receipt_returns_extended() {
        let p = ConnectionProfile::new();
        assert_eq!(p.header_for(FrameType::Receipt), HeaderFormat::Extended);
        assert_eq!(
            p.header_for(FrameType::EnergyGradient),
            HeaderFormat::Extended
        );
        assert_eq!(p.header_for(FrameType::Done), HeaderFormat::Extended);
    }

    #[test]
    fn batching_activates_after_enough_queries() {
        let mut p = ConnectionProfile::new();

        // 3 queries with 5 results each
        for _ in 0..3 {
            p.observe_query(100, false, 5);
        }

        assert!(
            p.optimal_batch_size > 1,
            "batch size should be > 1 after 3 queries with 5 results each, got {}",
            p.optimal_batch_size
        );
    }

    #[test]
    fn gradient_rising_triggers_throttle() {
        let mut p = ConnectionProfile::new();
        // Costs spike: 100, 200, 400, 800
        for cost in [100, 200, 400, 800] {
            p.observe_query(cost, false, 5);
        }
        assert!(p.energy_gradient() > 50, "gradient should be > 50");
        assert!(p.should_throttle());
        assert!(
            p.recommended_query_interval_ms > 100,
            "interval should increase with rising costs"
        );
    }

    #[test]
    fn gradient_falling_no_throttle() {
        let mut p = ConnectionProfile::new();
        // Costs drop: 800, 400, 200, 100
        for cost in [800, 400, 200, 100] {
            p.observe_query(cost, false, 5);
        }
        assert!(p.energy_gradient() < 0, "gradient should be negative");
        assert!(!p.should_throttle());
        assert_eq!(p.recommended_query_interval_ms, 100);
    }

    #[test]
    fn configurable_cache_hit_threshold() {
        let mut p = ConnectionProfile::new();
        assert_eq!(p.cache_hit_threshold_uwh, 50); // default

        p.cache_hit_threshold_uwh = 100;
        assert_eq!(p.cache_hit_threshold_uwh, 100);
    }

    #[test]
    fn rate_limit_update_and_check() {
        let mut p = ConnectionProfile::new();
        assert!(!p.is_rate_limited());
        assert!(p.rate_limit.is_none());

        // Server says 10 queries remaining, no retry
        p.update_rate_limit(&RateLimitPayload {
            queries_remaining: 10,
            window_seconds: 60,
            retry_after_ms: None,
            energy_budget_uwh: Some(5000),
        });
        assert!(!p.is_rate_limited());
        assert_eq!(p.rate_limit.as_ref().unwrap().queries_remaining, 10);

        // Server says 0 remaining → rate limited
        p.update_rate_limit(&RateLimitPayload {
            queries_remaining: 0,
            window_seconds: 60,
            retry_after_ms: None,
            energy_budget_uwh: None,
        });
        assert!(p.is_rate_limited());
    }

    #[test]
    fn rate_limit_retry_after() {
        let mut p = ConnectionProfile::new();

        p.update_rate_limit(&RateLimitPayload {
            queries_remaining: 5,
            window_seconds: 60,
            retry_after_ms: Some(2000),
            energy_budget_uwh: None,
        });
        assert!(p.is_rate_limited()); // retry_after_ms set → limited
    }

    #[test]
    fn cache_hit_rate_tracks() {
        let mut p = ConnectionProfile::new();

        // 4 queries: hit, miss, hit, hit
        p.observe_query(50, true, 5);
        p.observe_query(100, false, 5);
        p.observe_query(30, true, 5);
        p.observe_query(20, true, 5);

        // Should be > 5000 (above 50%) since 3/4 were hits
        assert!(
            p.cache_hit_rate > 5000,
            "cache_hit_rate should be > 5000, got {}",
            p.cache_hit_rate
        );
    }
}
