//! Reservoir sampling for metrics — uniform, exponentially decaying (biased
//! toward recent values), and sliding window reservoirs.
//!
//! Replaces `metrics-util` reservoir sampling with pure-Rust implementations
//! of Vitter's Algorithm R (uniform), forward-decay (exponential), and
//! sliding-window reservoirs with percentile computation and snapshots.

use std::fmt;

// ── Reservoir Snapshot ──────────────────────────────────────

/// An immutable snapshot of sampled values with percentile computation.
#[derive(Debug, Clone)]
pub struct ReservoirSnapshot {
    /// Sorted copy of the sampled values.
    values: Vec<f64>,
}

impl ReservoirSnapshot {
    /// Create from a slice (will be sorted internally).
    pub fn from_values(values: &[f64]) -> Self {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Self { values: sorted }
    }

    /// Number of values in the snapshot.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get the sorted values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Minimum value.
    pub fn min(&self) -> f64 {
        self.values.first().copied().unwrap_or(0.0)
    }

    /// Maximum value.
    pub fn max(&self) -> f64 {
        self.values.last().copied().unwrap_or(0.0)
    }

    /// Mean of all values.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Standard deviation.
    pub fn stddev(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = self.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (self.values.len() - 1) as f64;
        variance.sqrt()
    }

    /// Compute a percentile (0.0 to 100.0).
    pub fn percentile(&self, pct: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        if self.values.len() == 1 {
            return self.values[0];
        }
        let clamped = pct.clamp(0.0, 100.0);
        let rank = clamped / 100.0 * (self.values.len() - 1) as f64;
        let lower = rank.floor() as usize;
        let upper = (lower + 1).min(self.values.len() - 1);
        let fraction = rank - lower as f64;
        self.values[lower] + fraction * (self.values[upper] - self.values[lower])
    }

    /// Median (p50).
    pub fn median(&self) -> f64 {
        self.percentile(50.0)
    }

    /// 75th percentile.
    pub fn p75(&self) -> f64 {
        self.percentile(75.0)
    }

    /// 90th percentile.
    pub fn p90(&self) -> f64 {
        self.percentile(90.0)
    }

    /// 95th percentile.
    pub fn p95(&self) -> f64 {
        self.percentile(95.0)
    }

    /// 99th percentile.
    pub fn p99(&self) -> f64 {
        self.percentile(99.0)
    }

    /// 99.9th percentile.
    pub fn p999(&self) -> f64 {
        self.percentile(99.9)
    }
}

impl fmt::Display for ReservoirSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Snapshot(n={}, min={:.3}, max={:.3}, mean={:.3}, p50={:.3}, p99={:.3})",
            self.len(),
            self.min(),
            self.max(),
            self.mean(),
            self.median(),
            self.p99(),
        )
    }
}

// ── Simple PRNG ─────────────────────────────────────────────

/// A simple xorshift64 PRNG for reservoir sampling (no external dep).
#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Random f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Random index in [0, bound).
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

// ── Uniform Reservoir ───────────────────────────────────────

/// Uniform reservoir sampling (Vitter's Algorithm R).
/// Every element has an equal probability of being in the sample.
#[derive(Debug, Clone)]
pub struct UniformReservoir {
    /// Maximum number of samples to keep.
    capacity: usize,
    /// The sampled values.
    values: Vec<f64>,
    /// Total number of values ever added.
    total_count: u64,
    /// PRNG state.
    rng: Rng,
}

impl UniformReservoir {
    /// Create with the given capacity and a seed.
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self {
            capacity: capacity.max(1),
            values: Vec::with_capacity(capacity.max(1)),
            total_count: 0,
            rng: Rng::new(seed),
        }
    }

    /// Create with default seed.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(capacity, 42)
    }

    /// Add a value.
    pub fn update(&mut self, value: f64) {
        self.total_count += 1;
        if self.values.len() < self.capacity {
            self.values.push(value);
        } else {
            let idx = self.rng.next_usize(self.total_count as usize);
            if idx < self.capacity {
                self.values[idx] = value;
            }
        }
    }

    /// Number of samples currently stored.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total number of values ever added.
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Take a snapshot.
    pub fn snapshot(&self) -> ReservoirSnapshot {
        ReservoirSnapshot::from_values(&self.values)
    }

    /// Reset the reservoir.
    pub fn reset(&mut self) {
        self.values.clear();
        self.total_count = 0;
    }

    /// Get the raw sampled values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

// ── Exponentially Decaying Reservoir ────────────────────────

/// Entry in the decaying reservoir: priority + value.
#[derive(Debug, Clone)]
struct WeightedEntry {
    priority: f64,
    value: f64,
}

/// Exponentially decaying reservoir (forward-decay).
/// Biased toward recent values by exponentially decreasing the weight of
/// older entries. Uses Cormode et al.'s forward-decay priority scheme.
#[derive(Debug, Clone)]
pub struct ExponentiallyDecayingReservoir {
    capacity: usize,
    /// Decay factor (alpha). Higher = more bias toward recent.
    alpha: f64,
    /// Sorted entries by priority (descending — highest priority kept).
    entries: Vec<WeightedEntry>,
    /// Total count.
    total_count: u64,
    /// Start time for decay computation.
    start_time: f64,
    /// Next rescale time.
    next_rescale: f64,
    /// Rescale interval.
    rescale_interval: f64,
    rng: Rng,
}

impl ExponentiallyDecayingReservoir {
    /// Create with capacity, alpha (decay rate), and start time.
    pub fn new(capacity: usize, alpha: f64, start_time: f64) -> Self {
        let rescale_interval = 3600.0; // 1 hour
        Self {
            capacity: capacity.max(1),
            alpha,
            entries: Vec::with_capacity(capacity.max(1)),
            total_count: 0,
            start_time,
            next_rescale: start_time + rescale_interval,
            rescale_interval,
            rng: Rng::new(12345),
        }
    }

    /// Default: capacity=1028, alpha=0.015.
    pub fn default_reservoir(start_time: f64) -> Self {
        Self::new(1028, 0.015, start_time)
    }

    /// Add a value at the given time.
    pub fn update(&mut self, value: f64, time: f64) {
        self.total_count += 1;

        if time >= self.next_rescale {
            self.rescale(time);
        }

        let item_weight = (self.alpha * (time - self.start_time)).exp();
        let random_val = self.rng.next_f64().max(1e-18);
        let priority = item_weight / random_val;

        if self.entries.len() < self.capacity {
            self.entries.push(WeightedEntry { priority, value });
        } else {
            // Find entry with lowest priority
            let mut min_idx = 0;
            let mut min_pri = self.entries[0].priority;
            for (i, e) in self.entries.iter().enumerate().skip(1) {
                if e.priority < min_pri {
                    min_pri = e.priority;
                    min_idx = i;
                }
            }
            if priority > min_pri {
                self.entries[min_idx] = WeightedEntry { priority, value };
            }
        }
    }

    fn rescale(&mut self, now: f64) {
        let scale_factor = (-(self.alpha) * (now - self.start_time)).exp();
        self.start_time = now;
        self.next_rescale = now + self.rescale_interval;
        for entry in &mut self.entries {
            entry.priority *= scale_factor;
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Take a snapshot of the current values.
    pub fn snapshot(&self) -> ReservoirSnapshot {
        let values: Vec<f64> = self.entries.iter().map(|e| e.value).collect();
        ReservoirSnapshot::from_values(&values)
    }

    /// Reset the reservoir.
    pub fn reset(&mut self, new_start_time: f64) {
        self.entries.clear();
        self.total_count = 0;
        self.start_time = new_start_time;
        self.next_rescale = new_start_time + self.rescale_interval;
    }
}

// ── Sliding Window Reservoir ────────────────────────────────

/// A sliding window reservoir that keeps the most recent N values.
#[derive(Debug, Clone)]
pub struct SlidingWindowReservoir {
    capacity: usize,
    values: Vec<f64>,
    /// Write position (circular buffer index).
    write_pos: usize,
    /// Whether the buffer has wrapped around.
    wrapped: bool,
    total_count: u64,
}

impl SlidingWindowReservoir {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            capacity: cap,
            values: Vec::with_capacity(cap),
            write_pos: 0,
            wrapped: false,
            total_count: 0,
        }
    }

    /// Add a value.
    pub fn update(&mut self, value: f64) {
        self.total_count += 1;
        if self.values.len() < self.capacity {
            self.values.push(value);
        } else {
            self.values[self.write_pos] = value;
            self.wrapped = true;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Take a snapshot.
    pub fn snapshot(&self) -> ReservoirSnapshot {
        ReservoirSnapshot::from_values(&self.values)
    }

    /// Reset.
    pub fn reset(&mut self) {
        self.values.clear();
        self.write_pos = 0;
        self.wrapped = false;
        self.total_count = 0;
    }

    /// Get the values in insertion order (oldest first).
    pub fn values_ordered(&self) -> Vec<f64> {
        if !self.wrapped {
            return self.values.clone();
        }
        let mut result = Vec::with_capacity(self.capacity);
        result.extend_from_slice(&self.values[self.write_pos..]);
        result.extend_from_slice(&self.values[..self.write_pos]);
        result
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_empty() {
        let snap = ReservoirSnapshot::from_values(&[]);
        assert!(snap.is_empty());
        assert_eq!(snap.min(), 0.0);
        assert_eq!(snap.max(), 0.0);
        assert_eq!(snap.mean(), 0.0);
        assert_eq!(snap.median(), 0.0);
    }

    #[test]
    fn test_snapshot_single() {
        let snap = ReservoirSnapshot::from_values(&[42.0]);
        assert_eq!(snap.len(), 1);
        assert_eq!(snap.min(), 42.0);
        assert_eq!(snap.max(), 42.0);
        assert_eq!(snap.median(), 42.0);
    }

    #[test]
    fn test_snapshot_percentiles() {
        let vals: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let snap = ReservoirSnapshot::from_values(&vals);
        assert_eq!(snap.min(), 1.0);
        assert_eq!(snap.max(), 100.0);
        assert!((snap.mean() - 50.5).abs() < 0.01);
        assert!((snap.median() - 50.5).abs() < 0.01);
        assert!((snap.p75() - 75.25).abs() < 0.5);
        assert!((snap.p90() - 90.1).abs() < 0.5);
        assert!((snap.p95() - 95.05).abs() < 0.5);
        assert!((snap.p99() - 99.01).abs() < 0.5);
    }

    #[test]
    fn test_snapshot_stddev() {
        let snap = ReservoirSnapshot::from_values(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((snap.mean() - 5.0).abs() < 0.01);
        assert!(snap.stddev() > 1.0 && snap.stddev() < 3.0);
    }

    #[test]
    fn test_snapshot_display() {
        let snap = ReservoirSnapshot::from_values(&[1.0, 2.0, 3.0]);
        let text = format!("{}", snap);
        assert!(text.contains("n=3"));
        assert!(text.contains("min=1.000"));
    }

    #[test]
    fn test_uniform_reservoir_fill() {
        let mut r = UniformReservoir::with_capacity(10);
        for i in 0..10 {
            r.update(i as f64);
        }
        assert_eq!(r.len(), 10);
        assert_eq!(r.total_count(), 10);
    }

    #[test]
    fn test_uniform_reservoir_overflow() {
        let mut r = UniformReservoir::new(5, 42);
        for i in 0..100 {
            r.update(i as f64);
        }
        assert_eq!(r.len(), 5);
        assert_eq!(r.total_count(), 100);
    }

    #[test]
    fn test_uniform_reservoir_snapshot() {
        let mut r = UniformReservoir::with_capacity(100);
        for i in 1..=50 {
            r.update(i as f64);
        }
        let snap = r.snapshot();
        assert_eq!(snap.len(), 50);
        assert_eq!(snap.min(), 1.0);
        assert_eq!(snap.max(), 50.0);
    }

    #[test]
    fn test_uniform_reservoir_reset() {
        let mut r = UniformReservoir::with_capacity(10);
        for i in 0..10 {
            r.update(i as f64);
        }
        r.reset();
        assert!(r.is_empty());
        assert_eq!(r.total_count(), 0);
    }

    #[test]
    fn test_uniform_reservoir_capacity() {
        let r = UniformReservoir::with_capacity(50);
        assert_eq!(r.capacity(), 50);
    }

    #[test]
    fn test_exponential_reservoir_fill() {
        let mut r = ExponentiallyDecayingReservoir::new(100, 0.015, 0.0);
        for i in 0..50 {
            r.update(i as f64, i as f64);
        }
        assert_eq!(r.len(), 50);
        assert_eq!(r.total_count(), 50);
    }

    #[test]
    fn test_exponential_reservoir_capacity_limit() {
        let mut r = ExponentiallyDecayingReservoir::new(10, 0.015, 0.0);
        for i in 0..100 {
            r.update(i as f64, i as f64);
        }
        assert_eq!(r.len(), 10);
        assert_eq!(r.total_count(), 100);
    }

    #[test]
    fn test_exponential_reservoir_recent_bias() {
        let mut r = ExponentiallyDecayingReservoir::new(10, 1.0, 0.0);
        // Add old values
        for i in 0..100 {
            r.update(0.0, i as f64);
        }
        // Add recent values
        for i in 100..110 {
            r.update(100.0, i as f64);
        }
        // Recent values (100.0) should dominate due to high alpha
        let snap = r.snapshot();
        assert!(snap.mean() > 50.0, "mean={}", snap.mean());
    }

    #[test]
    fn test_exponential_reservoir_snapshot() {
        let mut r = ExponentiallyDecayingReservoir::default_reservoir(0.0);
        for i in 1..=20 {
            r.update(i as f64, i as f64);
        }
        let snap = r.snapshot();
        assert_eq!(snap.len(), 20);
    }

    #[test]
    fn test_exponential_reservoir_reset() {
        let mut r = ExponentiallyDecayingReservoir::new(10, 0.015, 0.0);
        r.update(1.0, 1.0);
        r.reset(100.0);
        assert!(r.is_empty());
        assert_eq!(r.total_count(), 0);
    }

    #[test]
    fn test_sliding_window_basic() {
        let mut r = SlidingWindowReservoir::new(5);
        for i in 1..=5 {
            r.update(i as f64);
        }
        assert_eq!(r.len(), 5);
        assert_eq!(r.total_count(), 5);
    }

    #[test]
    fn test_sliding_window_overflow() {
        let mut r = SlidingWindowReservoir::new(3);
        r.update(1.0);
        r.update(2.0);
        r.update(3.0);
        r.update(4.0); // overwrites 1.0
        r.update(5.0); // overwrites 2.0
        assert_eq!(r.len(), 3);
        assert_eq!(r.total_count(), 5);
        let snap = r.snapshot();
        assert_eq!(snap.min(), 3.0);
        assert_eq!(snap.max(), 5.0);
    }

    #[test]
    fn test_sliding_window_ordered() {
        let mut r = SlidingWindowReservoir::new(3);
        r.update(1.0);
        r.update(2.0);
        r.update(3.0);
        r.update(4.0);
        let ordered = r.values_ordered();
        assert_eq!(ordered, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sliding_window_reset() {
        let mut r = SlidingWindowReservoir::new(5);
        r.update(1.0);
        r.update(2.0);
        r.reset();
        assert!(r.is_empty());
        assert_eq!(r.total_count(), 0);
    }

    #[test]
    fn test_sliding_window_snapshot() {
        let mut r = SlidingWindowReservoir::new(100);
        for i in 1..=10 {
            r.update(i as f64);
        }
        let snap = r.snapshot();
        assert_eq!(snap.len(), 10);
        assert!((snap.mean() - 5.5).abs() < 0.01);
    }

    #[test]
    fn test_percentile_boundary() {
        let snap = ReservoirSnapshot::from_values(&[1.0, 2.0, 3.0]);
        assert_eq!(snap.percentile(0.0), 1.0);
        assert_eq!(snap.percentile(100.0), 3.0);
    }

    #[test]
    fn test_sliding_window_capacity() {
        let r = SlidingWindowReservoir::new(50);
        assert_eq!(r.capacity(), 50);
    }
}
