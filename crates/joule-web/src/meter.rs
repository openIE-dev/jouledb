//! Rate meter — events per second/minute/hour with EWMA (exponentially weighted
//! moving average) rates like Unix load averages.
//!
//! Replaces `metrics`, `cadence`, and `metered` rate meter implementations
//! with a pure-Rust tick-based meter supporting 1/5/15-minute EWMA rates,
//! mean rate, meter snapshot, and reset.

use std::fmt;

// ── Constants ───────────────────────────────────────────────

/// Tick interval in nanoseconds (5 seconds).
const TICK_INTERVAL_NS: u64 = 5_000_000_000;

/// Alpha for 1-minute EWMA (tick interval = 5s).
fn alpha_1min() -> f64 {
    1.0 - (-5.0_f64 / 60.0).exp()
}

/// Alpha for 5-minute EWMA.
fn alpha_5min() -> f64 {
    1.0 - (-5.0_f64 / 300.0).exp()
}

/// Alpha for 15-minute EWMA.
fn alpha_15min() -> f64 {
    1.0 - (-5.0_f64 / 900.0).exp()
}

// ── EWMA ────────────────────────────────────────────────────

/// Exponentially weighted moving average.
#[derive(Debug, Clone)]
pub struct Ewma {
    /// Smoothing factor.
    alpha: f64,
    /// Current rate in events per second.
    rate: f64,
    /// Uncounted events since last tick.
    uncounted: u64,
    /// Whether this EWMA has been initialized.
    initialized: bool,
    /// Tick interval in seconds.
    tick_interval_secs: f64,
}

impl Ewma {
    /// Create with a custom alpha and tick interval.
    pub fn new(alpha: f64, tick_interval_secs: f64) -> Self {
        Self {
            alpha,
            rate: 0.0,
            uncounted: 0,
            initialized: false,
            tick_interval_secs,
        }
    }

    /// 1-minute EWMA with 5-second tick interval.
    pub fn one_minute() -> Self {
        Self::new(alpha_1min(), 5.0)
    }

    /// 5-minute EWMA with 5-second tick interval.
    pub fn five_minute() -> Self {
        Self::new(alpha_5min(), 5.0)
    }

    /// 15-minute EWMA with 5-second tick interval.
    pub fn fifteen_minute() -> Self {
        Self::new(alpha_15min(), 5.0)
    }

    /// Mark N events.
    pub fn update(&mut self, n: u64) {
        self.uncounted += n;
    }

    /// Process one tick interval, updating the rate.
    pub fn tick(&mut self) {
        let instant_rate = self.uncounted as f64 / self.tick_interval_secs;
        self.uncounted = 0;
        if self.initialized {
            self.rate += self.alpha * (instant_rate - self.rate);
        } else {
            self.rate = instant_rate;
            self.initialized = true;
        }
    }

    /// Current rate in events per second.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Current rate in events per minute.
    pub fn rate_per_minute(&self) -> f64 {
        self.rate * 60.0
    }

    /// Current rate in events per hour.
    pub fn rate_per_hour(&self) -> f64 {
        self.rate * 3600.0
    }

    /// Whether this EWMA has received at least one tick.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.rate = 0.0;
        self.uncounted = 0;
        self.initialized = false;
    }
}

// ── Meter Snapshot ──────────────────────────────────────────

/// An immutable snapshot of a meter's state.
#[derive(Debug, Clone)]
pub struct MeterSnapshot {
    pub count: u64,
    pub mean_rate: f64,
    pub rate_1min: f64,
    pub rate_5min: f64,
    pub rate_15min: f64,
}

impl fmt::Display for MeterSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Meter(count={}, mean={:.3}/s, 1m={:.3}/s, 5m={:.3}/s, 15m={:.3}/s)",
            self.count, self.mean_rate, self.rate_1min, self.rate_5min, self.rate_15min,
        )
    }
}

// ── Meter ───────────────────────────────────────────────────

/// A rate meter that tracks events per second with 1/5/15-minute EWMA rates.
///
/// Call `tick()` every 5 seconds (the caller is responsible for scheduling).
/// Between ticks, use `mark(n)` to count events. The meter also tracks a
/// total count and a mean rate since creation.
#[derive(Debug, Clone)]
pub struct Meter {
    /// Total events counted.
    count: u64,
    /// Timestamp of meter creation (nanoseconds).
    start_ns: u64,
    /// 1-minute EWMA.
    m1: Ewma,
    /// 5-minute EWMA.
    m5: Ewma,
    /// 15-minute EWMA.
    m15: Ewma,
    /// Elapsed nanoseconds (updated on tick).
    elapsed_ns: u64,
}

impl Meter {
    /// Create a new meter with the given start time (nanoseconds).
    pub fn new(start_ns: u64) -> Self {
        Self {
            count: 0,
            start_ns,
            m1: Ewma::one_minute(),
            m5: Ewma::five_minute(),
            m15: Ewma::fifteen_minute(),
            elapsed_ns: 0,
        }
    }

    /// Create a meter starting at time 0.
    pub fn new_at_zero() -> Self {
        Self::new(0)
    }

    /// Mark a single event.
    pub fn mark(&mut self) {
        self.mark_n(1);
    }

    /// Mark N events.
    pub fn mark_n(&mut self, n: u64) {
        self.count += n;
        self.m1.update(n);
        self.m5.update(n);
        self.m15.update(n);
    }

    /// Advance by one tick interval (5 seconds). Call this periodically.
    pub fn tick(&mut self) {
        self.m1.tick();
        self.m5.tick();
        self.m15.tick();
        self.elapsed_ns += TICK_INTERVAL_NS;
    }

    /// Advance by multiple tick intervals at once.
    pub fn tick_n(&mut self, n: u32) {
        for _ in 0..n {
            self.tick();
        }
    }

    /// Total event count.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Mean rate since creation (events/second), based on elapsed ticks.
    pub fn mean_rate(&self) -> f64 {
        if self.elapsed_ns == 0 {
            return 0.0;
        }
        self.count as f64 / (self.elapsed_ns as f64 / 1_000_000_000.0)
    }

    /// 1-minute EWMA rate (events/second).
    pub fn rate_1min(&self) -> f64 {
        self.m1.rate()
    }

    /// 5-minute EWMA rate (events/second).
    pub fn rate_5min(&self) -> f64 {
        self.m5.rate()
    }

    /// 15-minute EWMA rate (events/second).
    pub fn rate_15min(&self) -> f64 {
        self.m15.rate()
    }

    /// 1-minute rate in events/minute.
    pub fn rate_1min_per_minute(&self) -> f64 {
        self.m1.rate_per_minute()
    }

    /// 5-minute rate in events/minute.
    pub fn rate_5min_per_minute(&self) -> f64 {
        self.m5.rate_per_minute()
    }

    /// 15-minute rate in events/minute.
    pub fn rate_15min_per_minute(&self) -> f64 {
        self.m15.rate_per_minute()
    }

    /// Take a snapshot of the current state.
    pub fn snapshot(&self) -> MeterSnapshot {
        MeterSnapshot {
            count: self.count,
            mean_rate: self.mean_rate(),
            rate_1min: self.m1.rate(),
            rate_5min: self.m5.rate(),
            rate_15min: self.m15.rate(),
        }
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.count = 0;
        self.elapsed_ns = 0;
        self.m1.reset();
        self.m5.reset();
        self.m15.reset();
    }

    /// The tick interval in nanoseconds.
    pub fn tick_interval_ns() -> u64 {
        TICK_INTERVAL_NS
    }
}

impl fmt::Display for Meter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.snapshot())
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_one_minute_initial() {
        let mut ewma = Ewma::one_minute();
        assert!(!ewma.is_initialized());
        ewma.update(100);
        ewma.tick();
        assert!(ewma.is_initialized());
        // Initial rate = 100 events / 5 seconds = 20/s
        assert!((ewma.rate() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_ewma_decay() {
        let mut ewma = Ewma::one_minute();
        ewma.update(100);
        ewma.tick();
        let initial = ewma.rate();
        // Tick without events -> rate should decay
        for _ in 0..12 {
            ewma.tick();
        }
        assert!(ewma.rate() < initial);
    }

    #[test]
    fn test_ewma_rate_per_minute() {
        let mut ewma = Ewma::one_minute();
        ewma.update(100);
        ewma.tick();
        assert!((ewma.rate_per_minute() - 20.0 * 60.0).abs() < 1.0);
    }

    #[test]
    fn test_ewma_rate_per_hour() {
        let mut ewma = Ewma::one_minute();
        ewma.update(100);
        ewma.tick();
        assert!((ewma.rate_per_hour() - 20.0 * 3600.0).abs() < 100.0);
    }

    #[test]
    fn test_ewma_reset() {
        let mut ewma = Ewma::one_minute();
        ewma.update(50);
        ewma.tick();
        ewma.reset();
        assert_eq!(ewma.rate(), 0.0);
        assert!(!ewma.is_initialized());
    }

    #[test]
    fn test_meter_mark() {
        let mut meter = Meter::new_at_zero();
        meter.mark();
        meter.mark();
        meter.mark_n(3);
        assert_eq!(meter.count(), 5);
    }

    #[test]
    fn test_meter_mean_rate_no_ticks() {
        let meter = Meter::new_at_zero();
        assert_eq!(meter.mean_rate(), 0.0);
    }

    #[test]
    fn test_meter_mean_rate() {
        let mut meter = Meter::new_at_zero();
        meter.mark_n(100);
        meter.tick(); // 5 seconds elapsed
        let rate = meter.mean_rate();
        assert!((rate - 20.0).abs() < 0.01, "mean_rate={}", rate);
    }

    #[test]
    fn test_meter_ewma_rates_after_tick() {
        let mut meter = Meter::new_at_zero();
        meter.mark_n(100);
        meter.tick();
        // After first tick: all rates should be ~20/s
        assert!((meter.rate_1min() - 20.0).abs() < 0.5);
        assert!((meter.rate_5min() - 20.0).abs() < 0.5);
        assert!((meter.rate_15min() - 20.0).abs() < 0.5);
    }

    #[test]
    fn test_meter_rates_decay() {
        let mut meter = Meter::new_at_zero();
        meter.mark_n(1000);
        meter.tick();
        let r1_initial = meter.rate_1min();
        // Tick many times with no new events
        for _ in 0..120 {
            meter.tick();
        }
        // 1-minute rate should have decayed significantly
        assert!(meter.rate_1min() < r1_initial * 0.01);
    }

    #[test]
    fn test_meter_15min_decays_slower() {
        let mut meter = Meter::new_at_zero();
        meter.mark_n(1000);
        meter.tick();
        // Tick with no events
        for _ in 0..12 {
            meter.tick();
        }
        // 15-min rate should be higher than 1-min rate (slower decay)
        assert!(
            meter.rate_15min() > meter.rate_1min(),
            "15m={} > 1m={}",
            meter.rate_15min(),
            meter.rate_1min()
        );
    }

    #[test]
    fn test_meter_per_minute_rates() {
        let mut meter = Meter::new_at_zero();
        meter.mark_n(60);
        meter.tick();
        // 60 events in 5 seconds = 12/s = 720/min
        let rpm = meter.rate_1min_per_minute();
        assert!((rpm - 720.0).abs() < 10.0, "rpm={}", rpm);
    }

    #[test]
    fn test_meter_snapshot() {
        let mut meter = Meter::new_at_zero();
        meter.mark_n(50);
        meter.tick();
        let snap = meter.snapshot();
        assert_eq!(snap.count, 50);
        assert!(snap.mean_rate > 0.0);
        assert!(snap.rate_1min > 0.0);
    }

    #[test]
    fn test_meter_snapshot_display() {
        let snap = MeterSnapshot {
            count: 100,
            mean_rate: 10.0,
            rate_1min: 9.5,
            rate_5min: 10.2,
            rate_15min: 10.1,
        };
        let text = format!("{}", snap);
        assert!(text.contains("count=100"));
        assert!(text.contains("mean=10.000"));
    }

    #[test]
    fn test_meter_reset() {
        let mut meter = Meter::new_at_zero();
        meter.mark_n(100);
        meter.tick();
        meter.reset();
        assert_eq!(meter.count(), 0);
        assert_eq!(meter.mean_rate(), 0.0);
        assert_eq!(meter.rate_1min(), 0.0);
    }

    #[test]
    fn test_meter_tick_n() {
        let mut meter = Meter::new_at_zero();
        meter.mark_n(50);
        meter.tick_n(3);
        // Should have advanced 15 seconds
        let rate = meter.mean_rate();
        let expected = 50.0 / 15.0;
        assert!((rate - expected).abs() < 0.01, "rate={}", rate);
    }

    #[test]
    fn test_meter_tick_interval() {
        assert_eq!(Meter::tick_interval_ns(), 5_000_000_000);
    }

    #[test]
    fn test_meter_display() {
        let mut meter = Meter::new_at_zero();
        meter.mark_n(10);
        meter.tick();
        let text = format!("{}", meter);
        assert!(text.contains("count=10"));
    }

    #[test]
    fn test_ewma_five_minute() {
        let mut ewma = Ewma::five_minute();
        ewma.update(100);
        ewma.tick();
        assert!((ewma.rate() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_ewma_fifteen_minute() {
        let mut ewma = Ewma::fifteen_minute();
        ewma.update(100);
        ewma.tick();
        assert!((ewma.rate() - 20.0).abs() < 0.01);
    }
}
