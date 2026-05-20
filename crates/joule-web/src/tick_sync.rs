//! Tick synchronization — clock synchronization across networked clients.
//!
//! Replaces custom tick sync in game engines with a pure-Rust system.
//! Maintains a local TickClock with configurable rate, adjusts it based on
//! server tick + RTT samples, detects and corrects drift, supports smooth
//! catchup (speed adjustment rather than jumps), and tracks ahead/behind
//! metrics.

use std::collections::VecDeque;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Tick synchronization errors.
#[derive(Debug, Clone, PartialEq)]
pub enum TickSyncError {
    /// Drift exceeds maximum tolerance.
    DriftExceeded { drift_ticks: f64, max: f64 },
    /// No RTT samples available.
    NoRttSamples,
    /// Invalid tick rate.
    InvalidTickRate { rate: f64 },
}

impl fmt::Display for TickSyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DriftExceeded { drift_ticks, max } => {
                write!(f, "drift {drift_ticks:.2} ticks exceeds max {max:.2}")
            }
            Self::NoRttSamples => write!(f, "no RTT samples available"),
            Self::InvalidTickRate { rate } => write!(f, "invalid tick rate: {rate}"),
        }
    }
}

impl std::error::Error for TickSyncError {}

// ── Tick Clock ──────────────────────────────────────────────────

/// A local tick clock with configurable rate.
#[derive(Debug, Clone)]
pub struct TickClock {
    tick: u64,
    fractional: f64,
    tick_rate: f64,
    tick_interval: f64,
    speed_multiplier: f64,
    accumulated_time: f64,
}

impl TickClock {
    pub fn new(tick_rate: f64) -> Self {
        Self {
            tick: 0,
            fractional: 0.0,
            tick_rate,
            tick_interval: 1.0 / tick_rate,
            speed_multiplier: 1.0,
            accumulated_time: 0.0,
        }
    }

    /// Advance the clock by dt seconds. Returns how many ticks elapsed.
    pub fn advance(&mut self, dt: f64) -> u32 {
        self.accumulated_time += dt * self.speed_multiplier;
        let mut ticks_elapsed = 0u32;

        while self.accumulated_time >= self.tick_interval {
            self.accumulated_time -= self.tick_interval;
            self.tick += 1;
            ticks_elapsed += 1;
        }
        self.fractional = self.accumulated_time / self.tick_interval;
        ticks_elapsed
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn precise_tick(&self) -> f64 {
        self.tick as f64 + self.fractional
    }

    pub fn tick_rate(&self) -> f64 {
        self.tick_rate
    }

    pub fn speed_multiplier(&self) -> f64 {
        self.speed_multiplier
    }

    pub fn set_speed_multiplier(&mut self, m: f64) {
        self.speed_multiplier = m.clamp(0.5, 2.0);
    }

    pub fn set_tick(&mut self, tick: u64) {
        self.tick = tick;
        self.fractional = 0.0;
        self.accumulated_time = 0.0;
    }
}

impl fmt::Display for TickClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TickClock(tick={}, rate={:.0}Hz, speed={:.2}x)", self.tick, self.tick_rate, self.speed_multiplier)
    }
}

// ── RTT Tracker ─────────────────────────────────────────────────

/// Tracks round-trip time samples.
#[derive(Debug)]
pub struct RttTracker {
    samples: VecDeque<f64>,
    max_samples: usize,
    total: f64,
}

impl RttTracker {
    pub fn new(max_samples: usize) -> Self {
        Self { samples: VecDeque::with_capacity(max_samples), max_samples, total: 0.0 }
    }

    pub fn record(&mut self, rtt: f64) {
        if self.samples.len() >= self.max_samples {
            if let Some(old) = self.samples.pop_front() {
                self.total -= old;
            }
        }
        self.total += rtt;
        self.samples.push_back(rtt);
    }

    pub fn average(&self) -> Option<f64> {
        if self.samples.is_empty() {
            None
        } else {
            Some(self.total / self.samples.len() as f64)
        }
    }

    pub fn min(&self) -> Option<f64> {
        self.samples.iter().cloned().reduce(f64::min)
    }

    pub fn max(&self) -> Option<f64> {
        self.samples.iter().cloned().reduce(f64::max)
    }

    pub fn jitter(&self) -> Option<f64> {
        let avg = self.average()?;
        let variance = self.samples.iter().map(|s| (s - avg).powi(2)).sum::<f64>()
            / self.samples.len() as f64;
        Some(variance.sqrt())
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn clear(&mut self) {
        self.samples.clear();
        self.total = 0.0;
    }
}

// ── Drift Detector ──────────────────────────────────────────────

/// Detects clock drift between local and server ticks.
#[derive(Debug)]
pub struct DriftDetector {
    drift_history: VecDeque<f64>,
    max_history: usize,
}

impl DriftDetector {
    pub fn new(max_history: usize) -> Self {
        Self { drift_history: VecDeque::with_capacity(max_history), max_history }
    }

    /// Record a drift measurement (local_tick - expected_tick).
    pub fn record(&mut self, drift: f64) {
        if self.drift_history.len() >= self.max_history {
            self.drift_history.pop_front();
        }
        self.drift_history.push_back(drift);
    }

    /// Average drift over the window.
    pub fn average_drift(&self) -> f64 {
        if self.drift_history.is_empty() {
            return 0.0;
        }
        self.drift_history.iter().sum::<f64>() / self.drift_history.len() as f64
    }

    /// Trend: positive = drifting ahead, negative = falling behind.
    pub fn drift_trend(&self) -> f64 {
        if self.drift_history.len() < 2 {
            return 0.0;
        }
        let n = self.drift_history.len();
        let first_half: f64 =
            self.drift_history.iter().take(n / 2).sum::<f64>() / (n / 2) as f64;
        let second_half: f64 =
            self.drift_history.iter().skip(n / 2).sum::<f64>() / (n - n / 2) as f64;
        second_half - first_half
    }
}

// ── Sync Metrics ────────────────────────────────────────────────

/// Runtime metrics for tick synchronization.
#[derive(Debug, Clone, Default)]
pub struct SyncMetrics {
    pub sync_count: u64,
    pub total_corrections: u64,
    pub max_drift_observed: f64,
    pub speedup_periods: u64,
    pub slowdown_periods: u64,
    pub hard_resets: u64,
}

impl fmt::Display for SyncMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Syncs: {}, Corrections: {}, MaxDrift: {:.2}, Speedups: {}, Slowdowns: {}",
            self.sync_count, self.total_corrections, self.max_drift_observed,
            self.speedup_periods, self.slowdown_periods
        )
    }
}

// ── Tick Sync Config ────────────────────────────────────────────

/// Configuration for tick synchronization.
#[derive(Debug, Clone)]
pub struct TickSyncConfig {
    pub tick_rate: f64,
    pub max_drift_tolerance: f64,
    pub smooth_factor: f64,
    pub rtt_samples: usize,
    pub drift_history: usize,
    pub hard_reset_threshold: f64,
    pub speed_adjust_factor: f64,
}

impl TickSyncConfig {
    pub fn new(tick_rate: f64) -> Self {
        Self {
            tick_rate,
            max_drift_tolerance: 5.0,
            smooth_factor: 0.1,
            rtt_samples: 32,
            drift_history: 64,
            hard_reset_threshold: 30.0,
            speed_adjust_factor: 0.02,
        }
    }

    pub fn with_max_drift(mut self, ticks: f64) -> Self {
        self.max_drift_tolerance = ticks;
        self
    }

    pub fn with_smooth_factor(mut self, f: f64) -> Self {
        self.smooth_factor = f;
        self
    }

    pub fn with_hard_reset_threshold(mut self, ticks: f64) -> Self {
        self.hard_reset_threshold = ticks;
        self
    }
}

impl Default for TickSyncConfig {
    fn default() -> Self {
        Self::new(60.0)
    }
}

// ── Tick Sync ───────────────────────────────────────────────────

/// Main tick synchronization engine.
#[derive(Debug)]
pub struct TickSync {
    config: TickSyncConfig,
    clock: TickClock,
    rtt: RttTracker,
    drift: DriftDetector,
    metrics: SyncMetrics,
    last_server_tick: u64,
}

impl TickSync {
    pub fn new(config: TickSyncConfig) -> Self {
        let tick_rate = config.tick_rate;
        let rtt_cap = config.rtt_samples;
        let drift_cap = config.drift_history;
        Self {
            config,
            clock: TickClock::new(tick_rate),
            rtt: RttTracker::new(rtt_cap),
            drift: DriftDetector::new(drift_cap),
            metrics: SyncMetrics::default(),
            last_server_tick: 0,
        }
    }

    /// Advance the local clock by dt seconds.
    pub fn advance(&mut self, dt: f64) -> u32 {
        self.clock.advance(dt)
    }

    /// Process a server tick update with RTT.
    pub fn on_server_tick(&mut self, server_tick: u64, rtt_seconds: f64) {
        self.rtt.record(rtt_seconds);
        self.metrics.sync_count += 1;
        self.last_server_tick = server_tick;

        // Estimate what our tick should be: server_tick + half_rtt_in_ticks.
        let half_rtt_ticks = (rtt_seconds / 2.0) * self.config.tick_rate;
        let expected_local = server_tick as f64 + half_rtt_ticks;
        let actual_local = self.clock.precise_tick();
        let drift = actual_local - expected_local;

        self.drift.record(drift);

        let abs_drift = drift.abs();
        if abs_drift > self.metrics.max_drift_observed {
            self.metrics.max_drift_observed = abs_drift;
        }

        // Hard reset if drift is extreme.
        if abs_drift > self.config.hard_reset_threshold {
            self.clock.set_tick(expected_local.round() as u64);
            self.clock.set_speed_multiplier(1.0);
            self.metrics.hard_resets += 1;
            self.metrics.total_corrections += 1;
            return;
        }

        // Smooth correction: adjust speed multiplier.
        if abs_drift > self.config.max_drift_tolerance {
            self.metrics.total_corrections += 1;
            if drift > 0.0 {
                // Running ahead — slow down.
                let slowdown = 1.0 - self.config.speed_adjust_factor * (abs_drift / self.config.max_drift_tolerance);
                self.clock.set_speed_multiplier(slowdown.max(0.5));
                self.metrics.slowdown_periods += 1;
            } else {
                // Running behind — speed up.
                let speedup = 1.0 + self.config.speed_adjust_factor * (abs_drift / self.config.max_drift_tolerance);
                self.clock.set_speed_multiplier(speedup.min(2.0));
                self.metrics.speedup_periods += 1;
            }
        } else {
            // Within tolerance — gradually return to normal speed.
            let current = self.clock.speed_multiplier();
            let target = 1.0;
            let smoothed = current + (target - current) * self.config.smooth_factor;
            self.clock.set_speed_multiplier(smoothed);
        }
    }

    /// Current drift in ticks (positive = ahead, negative = behind).
    pub fn current_drift(&self) -> f64 {
        self.drift.average_drift()
    }

    /// Drift trend (positive = increasingly ahead).
    pub fn drift_trend(&self) -> f64 {
        self.drift.drift_trend()
    }

    pub fn local_tick(&self) -> u64 {
        self.clock.tick()
    }

    pub fn precise_tick(&self) -> f64 {
        self.clock.precise_tick()
    }

    pub fn server_tick(&self) -> u64 {
        self.last_server_tick
    }

    pub fn average_rtt(&self) -> Option<f64> {
        self.rtt.average()
    }

    pub fn rtt_jitter(&self) -> Option<f64> {
        self.rtt.jitter()
    }

    pub fn speed_multiplier(&self) -> f64 {
        self.clock.speed_multiplier()
    }

    pub fn metrics(&self) -> &SyncMetrics {
        &self.metrics
    }

    pub fn clock(&self) -> &TickClock {
        &self.clock
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_clock_advance() {
        let mut clock = TickClock::new(60.0);
        let ticks = clock.advance(1.0 / 60.0);
        assert_eq!(ticks, 1);
        assert_eq!(clock.tick(), 1);
    }

    #[test]
    fn tick_clock_multiple_ticks() {
        let mut clock = TickClock::new(60.0);
        let ticks = clock.advance(0.5); // 30 ticks
        assert_eq!(ticks, 30);
        assert_eq!(clock.tick(), 30);
    }

    #[test]
    fn tick_clock_speed_multiplier() {
        let mut clock = TickClock::new(60.0);
        clock.set_speed_multiplier(2.0);
        let ticks = clock.advance(1.0 / 60.0); // should get ~2 ticks
        assert_eq!(ticks, 2);
    }

    #[test]
    fn tick_clock_speed_clamped() {
        let mut clock = TickClock::new(60.0);
        clock.set_speed_multiplier(10.0);
        assert!((clock.speed_multiplier() - 2.0).abs() < 1e-9);
        clock.set_speed_multiplier(-1.0);
        assert!((clock.speed_multiplier() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn tick_clock_precise() {
        let mut clock = TickClock::new(60.0);
        clock.advance(0.5 / 60.0); // half a tick
        assert!(clock.precise_tick() > 0.0);
        assert!(clock.precise_tick() < 1.0);
    }

    #[test]
    fn tick_clock_display() {
        let clock = TickClock::new(60.0);
        let s = format!("{clock}");
        assert!(s.contains("rate=60Hz"));
    }

    #[test]
    fn rtt_tracker_average() {
        let mut rtt = RttTracker::new(10);
        rtt.record(0.050);
        rtt.record(0.060);
        let avg = rtt.average().unwrap();
        assert!((avg - 0.055).abs() < 1e-9);
    }

    #[test]
    fn rtt_tracker_jitter() {
        let mut rtt = RttTracker::new(10);
        rtt.record(0.050);
        rtt.record(0.050);
        let jitter = rtt.jitter().unwrap();
        assert!(jitter < 1e-9); // no jitter
    }

    #[test]
    fn rtt_tracker_min_max() {
        let mut rtt = RttTracker::new(10);
        rtt.record(0.050);
        rtt.record(0.020);
        rtt.record(0.080);
        assert!((rtt.min().unwrap() - 0.020).abs() < 1e-9);
        assert!((rtt.max().unwrap() - 0.080).abs() < 1e-9);
    }

    #[test]
    fn drift_detector_average() {
        let mut det = DriftDetector::new(10);
        det.record(2.0);
        det.record(4.0);
        assert!((det.average_drift() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn drift_detector_trend() {
        let mut det = DriftDetector::new(10);
        det.record(1.0);
        det.record(1.0);
        det.record(3.0);
        det.record(3.0);
        let trend = det.drift_trend();
        assert!(trend > 0.0); // drifting further ahead
    }

    #[test]
    fn tick_sync_basic_advance() {
        let config = TickSyncConfig::new(60.0);
        let mut sync = TickSync::new(config);
        // Advance in individual ticks to avoid fp accumulation drift.
        for _ in 0..60 {
            sync.advance(1.0 / 60.0);
        }
        assert_eq!(sync.local_tick(), 60);
    }

    #[test]
    fn tick_sync_server_update() {
        let config = TickSyncConfig::new(60.0);
        let mut sync = TickSync::new(config);
        for _ in 0..60 {
            sync.advance(1.0 / 60.0);
        }
        sync.on_server_tick(60, 0.050);
        assert_eq!(sync.server_tick(), 60);
        assert!(sync.average_rtt().is_some());
    }

    #[test]
    fn tick_sync_speed_adjustment_ahead() {
        let config = TickSyncConfig::new(60.0).with_max_drift(2.0).with_hard_reset_threshold(1000.0);
        let mut sync = TickSync::new(config);
        // Advance far ahead of server.
        for _ in 0..120 {
            sync.advance(1.0 / 60.0);
        }
        sync.on_server_tick(10, 0.010); // server way behind, drift > tolerance
        assert!(sync.speed_multiplier() < 1.0, "multiplier={}", sync.speed_multiplier());
    }

    #[test]
    fn tick_sync_speed_adjustment_behind() {
        let config = TickSyncConfig::new(60.0).with_max_drift(2.0).with_hard_reset_threshold(1000.0);
        let mut sync = TickSync::new(config);
        // Don't advance, but server is ahead — drift is negative (behind).
        sync.on_server_tick(100, 0.010);
        assert!(sync.speed_multiplier() > 1.0, "multiplier={}", sync.speed_multiplier());
    }

    #[test]
    fn tick_sync_hard_reset() {
        let config = TickSyncConfig::new(60.0).with_hard_reset_threshold(10.0);
        let mut sync = TickSync::new(config);
        // Server says tick 1000, we're at 0.
        sync.on_server_tick(1000, 0.010);
        assert_eq!(sync.metrics().hard_resets, 1);
        // Should be close to 1000 now.
        assert!(sync.local_tick() > 900);
    }

    #[test]
    fn tick_sync_metrics_display() {
        let m = SyncMetrics {
            sync_count: 100,
            total_corrections: 5,
            max_drift_observed: 3.5,
            speedup_periods: 3,
            slowdown_periods: 2,
            hard_resets: 0,
        };
        let s = format!("{m}");
        assert!(s.contains("Syncs: 100"));
    }

    #[test]
    fn tick_sync_within_tolerance() {
        let config = TickSyncConfig::new(60.0).with_max_drift(100.0);
        let mut sync = TickSync::new(config);
        sync.advance(1.0 / 60.0);
        sync.on_server_tick(1, 0.001);
        // Within tolerance, speed should be near 1.0.
        assert!((sync.speed_multiplier() - 1.0).abs() < 0.1);
    }
}
