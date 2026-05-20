//! Deterministic loop timing with jitter and drift measurement.
//!
//! The [`LoopTimer`] replaces raw `tokio::time::sleep` / `tokio::time::interval`
//! in background loops, providing:
//!
//! 1. **Jitter** — randomized offset (±10% default) to prevent thundering herd
//!    when multiple nodes/loops share the same base interval.
//! 2. **`MissedTickBehavior::Delay`** — prevents burst catch-up after GC pauses
//!    or heavy CPU load. Required for deterministic timing in auditable systems.
//! 3. **Drift measurement** — tracks actual-vs-expected tick latency.
//! 4. **Startup stagger** — optional initial delay to spread loop starts.
//!
//! Regulatory basis:
//! - NIST SI-4 (system monitoring precision)
//! - DORA Art 10 (detection timeliness)
//! - ISO 27001 A.8.16 (monitoring activities)

use std::time::{Duration, Instant};

use tokio::time::{Interval, MissedTickBehavior, interval};

/// A loop timer with jitter, drift tracking, and missed-tick protection.
pub struct LoopTimer {
    /// The configured base interval.
    base_interval: Duration,
    /// The tokio interval (with MissedTickBehavior::Delay).
    interval: Interval,
    /// Name of this loop (for metrics/logging).
    name: String,
    /// Running count of ticks.
    tick_count: u64,
    /// Maximum observed drift from the expected tick time.
    max_drift_ms: f64,
    /// Sum of drift values (for computing average).
    total_drift_ms: f64,
    /// Last tick wall-clock time.
    last_tick: Option<Instant>,
}

impl LoopTimer {
    /// Create a new loop timer with the given name and base interval.
    ///
    /// Applies ±10% jitter to the base interval and sets `MissedTickBehavior::Delay`.
    pub fn new(name: impl Into<String>, base_interval: Duration) -> Self {
        let jittered = apply_jitter(base_interval, 0.10);
        let mut interval = interval(jittered);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

        Self {
            base_interval,
            interval,
            name: name.into(),
            tick_count: 0,
            max_drift_ms: 0.0,
            total_drift_ms: 0.0,
            last_tick: None,
        }
    }

    /// Create with a custom jitter fraction (0.0 = no jitter, 0.20 = ±20%).
    pub fn with_jitter(name: impl Into<String>, base_interval: Duration, jitter_frac: f64) -> Self {
        let jittered = apply_jitter(base_interval, jitter_frac);
        let mut interval = interval(jittered);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

        Self {
            base_interval,
            interval,
            name: name.into(),
            tick_count: 0,
            max_drift_ms: 0.0,
            total_drift_ms: 0.0,
            last_tick: None,
        }
    }

    /// Wait for the next tick. Returns drift information for this tick.
    pub async fn tick(&mut self) -> TickInfo {
        self.interval.tick().await;
        let now = Instant::now();

        let drift_ms = if let Some(last) = self.last_tick {
            let actual_ms = now.duration_since(last).as_secs_f64() * 1000.0;
            let expected_ms = self.base_interval.as_secs_f64() * 1000.0;
            (actual_ms - expected_ms).abs()
        } else {
            0.0
        };

        self.last_tick = Some(now);
        self.tick_count += 1;
        self.total_drift_ms += drift_ms;
        if drift_ms > self.max_drift_ms {
            self.max_drift_ms = drift_ms;
        }

        TickInfo {
            tick_number: self.tick_count,
            drift_ms,
        }
    }

    /// The loop name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Total ticks since creation.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Average drift across all ticks (milliseconds).
    pub fn avg_drift_ms(&self) -> f64 {
        if self.tick_count == 0 {
            0.0
        } else {
            self.total_drift_ms / self.tick_count as f64
        }
    }

    /// Maximum observed drift (milliseconds).
    pub fn max_drift_ms(&self) -> f64 {
        self.max_drift_ms
    }

    /// The base interval.
    pub fn base_interval(&self) -> Duration {
        self.base_interval
    }
}

/// Information about a single tick.
#[derive(Debug, Clone, Copy)]
pub struct TickInfo {
    /// The tick number (1-based).
    pub tick_number: u64,
    /// Drift from expected interval in milliseconds.
    pub drift_ms: f64,
}

/// Apply jitter to a duration: returns a duration uniformly sampled from
/// `[base * (1 - frac), base * (1 + frac)]`.
fn apply_jitter(base: Duration, frac: f64) -> Duration {
    if frac <= 0.0 || frac >= 1.0 {
        return base;
    }
    let base_ms = base.as_millis() as f64;
    // Use a simple hash of the current time as entropy (no rand crate needed)
    let entropy = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    // Map to [0, 1)
    let r = (entropy as f64) / (u32::MAX as f64);
    // Map to [-frac, +frac]
    let offset = base_ms * frac * (2.0 * r - 1.0);
    let jittered_ms = (base_ms + offset).max(1.0);
    Duration::from_millis(jittered_ms as u64)
}

/// Sleep for a staggered startup delay based on a loop index.
///
/// Spreads `count` loops across `spread` duration, starting at offset 0.
pub async fn stagger_start(index: usize, count: usize, spread: Duration) {
    if count <= 1 {
        return;
    }
    let step = spread.as_millis() as f64 / count as f64;
    let delay = Duration::from_millis((step * index as f64) as u64);
    tokio::time::sleep(delay).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_jitter_no_frac() {
        let base = Duration::from_secs(10);
        let result = apply_jitter(base, 0.0);
        assert_eq!(result, base);
    }

    #[test]
    fn apply_jitter_within_bounds() {
        let base = Duration::from_secs(10);
        let result = apply_jitter(base, 0.10);
        let base_ms = base.as_millis();
        let result_ms = result.as_millis();
        // Should be within ±10% of base
        assert!(result_ms >= base_ms * 90 / 100);
        assert!(result_ms <= base_ms * 110 / 100);
    }

    #[test]
    fn apply_jitter_frac_1_returns_base() {
        let base = Duration::from_secs(5);
        let result = apply_jitter(base, 1.0);
        assert_eq!(result, base);
    }

    #[tokio::test]
    async fn loop_timer_ticks() {
        let mut timer = LoopTimer::new("test", Duration::from_millis(10));
        let info = timer.tick().await;
        assert_eq!(info.tick_number, 1);
        assert_eq!(timer.tick_count(), 1);
    }

    #[tokio::test]
    async fn loop_timer_tracks_drift() {
        let mut timer = LoopTimer::new("drift-test", Duration::from_millis(20));
        // First tick has no drift reference
        timer.tick().await;
        assert_eq!(timer.tick_count(), 1);

        // Second tick measures drift
        timer.tick().await;
        assert_eq!(timer.tick_count(), 2);
        // Drift should be small (timer is reasonably accurate)
        assert!(timer.max_drift_ms() < 100.0, "drift should be small");
    }

    #[tokio::test]
    async fn stagger_start_single() {
        // Single loop should not sleep
        let start = Instant::now();
        stagger_start(0, 1, Duration::from_secs(1)).await;
        assert!(start.elapsed() < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn stagger_start_spreads() {
        let start = Instant::now();
        stagger_start(0, 4, Duration::from_millis(100)).await;
        // Index 0 should be immediate
        assert!(start.elapsed() < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn loop_timer_name() {
        let timer = LoopTimer::new("heartbeat", Duration::from_secs(30));
        assert_eq!(timer.name(), "heartbeat");
        assert_eq!(timer.base_interval(), Duration::from_secs(30));
    }

    #[tokio::test]
    async fn avg_drift_zero_ticks() {
        let timer = LoopTimer::new("empty", Duration::from_secs(1));
        assert!((timer.avg_drift_ms() - 0.0).abs() < f64::EPSILON);
    }
}
