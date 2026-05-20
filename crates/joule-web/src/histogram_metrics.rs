//! Histogram implementation for metrics — configurable buckets, HDR histogram,
//! percentile estimation, and histogram merge.
//!
//! Replaces `hdrhistogram`, `metrics-util`, and `histogram` crates with a
//! pure-Rust implementation supporting both linear/exponential bucket histograms
//! and a high dynamic range (HDR) histogram with configurable precision.

use std::fmt;

// ── Bucket Configuration ────────────────────────────────────

/// How to configure histogram buckets.
#[derive(Debug, Clone, PartialEq)]
pub enum BucketConfig {
    /// Explicit upper bounds.
    Explicit(Vec<f64>),
    /// Linear: start, width, count.
    Linear {
        start: f64,
        width: f64,
        count: usize,
    },
    /// Exponential: start, factor, count.
    Exponential {
        start: f64,
        factor: f64,
        count: usize,
    },
}

impl BucketConfig {
    /// Generate the upper bounds from this config.
    pub fn bounds(&self) -> Vec<f64> {
        match self {
            BucketConfig::Explicit(v) => {
                let mut sorted = v.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                sorted
            }
            BucketConfig::Linear {
                start,
                width,
                count,
            } => (0..*count).map(|i| start + width * i as f64).collect(),
            BucketConfig::Exponential {
                start,
                factor,
                count,
            } => (0..*count).map(|i| start * factor.powi(i as i32)).collect(),
        }
    }
}

// ── Bucket Histogram ────────────────────────────────────────

/// A histogram with fixed upper-bound buckets.
#[derive(Debug, Clone)]
pub struct BucketHistogram {
    /// Upper bounds for each bucket.
    bounds: Vec<f64>,
    /// Count of observations in each bucket (not cumulative).
    counts: Vec<u64>,
    /// Sum of all observed values.
    sum: f64,
    /// Total observation count.
    count: u64,
    /// Minimum observed value.
    min: f64,
    /// Maximum observed value.
    max: f64,
}

impl BucketHistogram {
    /// Create from explicit upper bounds.
    pub fn new(bounds: &[f64]) -> Self {
        let mut sorted = bounds.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted.dedup();
        let len = sorted.len();
        Self {
            bounds: sorted,
            counts: vec![0; len + 1], // +1 for overflow bucket
            sum: 0.0,
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Create from a BucketConfig.
    pub fn from_config(config: &BucketConfig) -> Self {
        Self::new(&config.bounds())
    }

    /// Record a value.
    pub fn observe(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        // Find the first bucket where value <= bound
        let mut placed = false;
        for (i, bound) in self.bounds.iter().enumerate() {
            if value <= *bound {
                self.counts[i] += 1;
                placed = true;
                break;
            }
        }
        if !placed {
            // Overflow bucket
            let last = self.counts.len() - 1;
            self.counts[last] += 1;
        }
    }

    pub fn sum(&self) -> f64 {
        self.sum
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn min(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.min
        }
    }

    pub fn max(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.max
        }
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Get the upper bounds.
    pub fn bounds(&self) -> &[f64] {
        &self.bounds
    }

    /// Get the per-bucket counts (including overflow bucket at the end).
    pub fn bucket_counts(&self) -> &[u64] {
        &self.counts
    }

    /// Get cumulative counts (each bucket includes all previous).
    pub fn cumulative_counts(&self) -> Vec<u64> {
        let mut result = Vec::with_capacity(self.counts.len());
        let mut acc = 0u64;
        for c in &self.counts {
            acc += c;
            result.push(acc);
        }
        result
    }

    /// Estimate a percentile (0.0 to 100.0) from bucket data.
    /// Uses linear interpolation within the bucket.
    pub fn percentile(&self, pct: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let target = (pct / 100.0 * self.count as f64).ceil() as u64;
        let target = target.max(1).min(self.count);

        let mut cumulative = 0u64;
        for (i, c) in self.counts.iter().enumerate() {
            let prev_cumulative = cumulative;
            cumulative += c;
            if cumulative >= target && *c > 0 {
                let lower = if i == 0 {
                    0.0
                } else if i <= self.bounds.len() {
                    self.bounds[i.saturating_sub(1)]
                } else {
                    *self.bounds.last().unwrap_or(&0.0)
                };
                let upper = if i < self.bounds.len() {
                    self.bounds[i]
                } else {
                    self.max
                };
                // Interpolate within bucket
                let rank_in_bucket = target - prev_cumulative;
                let fraction = rank_in_bucket as f64 / *c as f64;
                return lower + (upper - lower) * fraction;
            }
        }
        self.max
    }

    /// Merge another histogram (must have identical bounds).
    pub fn merge(&mut self, other: &BucketHistogram) -> Result<(), &'static str> {
        if self.bounds != other.bounds {
            return Err("cannot merge histograms with different bounds");
        }
        for (i, c) in other.counts.iter().enumerate() {
            self.counts[i] += c;
        }
        self.sum += other.sum;
        self.count += other.count;
        if other.count > 0 {
            if other.min < self.min {
                self.min = other.min;
            }
            if other.max > self.max {
                self.max = other.max;
            }
        }
        Ok(())
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        for c in &mut self.counts {
            *c = 0;
        }
        self.sum = 0.0;
        self.count = 0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }
}

// ── HDR Histogram ───────────────────────────────────────────

/// High Dynamic Range (HDR) histogram.
///
/// Records values from 1 to `max_value` with a given number of significant
/// digits of precision. Uses a logarithmic + linear sub-bucket scheme.
#[derive(Debug, Clone)]
pub struct HdrHistogram {
    /// Maximum trackable value.
    max_trackable: u64,
    /// Number of significant digits (1-5).
    significant_digits: u8,
    /// Sub-bucket count (derived from significant digits).
    sub_bucket_count: u32,
    /// Sub-bucket half count.
    sub_bucket_half_count: u32,
    /// Bit length of sub_bucket_count.
    sub_bucket_mask: u32,
    /// Number of magnitude levels (bucket count).
    bucket_count: u32,
    /// Counts array.
    counts: Vec<u64>,
    /// Total count.
    total_count: u64,
    /// Min recorded value.
    min_value: u64,
    /// Max recorded value.
    max_value: u64,
}

impl HdrHistogram {
    /// Create a new HDR histogram.
    ///
    /// - `max_trackable`: the maximum value this histogram can record.
    /// - `significant_digits`: number of significant digits of precision (1-5).
    pub fn new(max_trackable: u64, significant_digits: u8) -> Self {
        let sig = significant_digits.min(5).max(1);
        let largest_exact = 2u64 * 10u64.pow(sig as u32);
        let sub_bucket_count_magnitude = (largest_exact as f64).log2().ceil() as u32;
        let sub_bucket_count = 1u32 << sub_bucket_count_magnitude;
        let sub_bucket_half_count = sub_bucket_count >> 1;
        let sub_bucket_mask = sub_bucket_count - 1;

        // Calculate bucket count
        let mut bucket_count = 1u32;
        let mut smallest_untrackable = (sub_bucket_count as u64) << 1;
        while smallest_untrackable <= max_trackable {
            smallest_untrackable <<= 1;
            bucket_count += 1;
        }

        let counts_len = (bucket_count as usize + 1) * sub_bucket_half_count as usize + 1;

        Self {
            max_trackable,
            significant_digits: sig,
            sub_bucket_count,
            sub_bucket_half_count,
            sub_bucket_mask,
            bucket_count,
            counts: vec![0; counts_len],
            total_count: 0,
            min_value: u64::MAX,
            max_value: 0,
        }
    }

    /// Record a value.
    pub fn record(&mut self, value: u64) -> bool {
        if value > self.max_trackable {
            return false;
        }
        let idx = self.counts_index(value);
        if idx >= self.counts.len() {
            return false;
        }
        self.counts[idx] += 1;
        self.total_count += 1;
        if value < self.min_value {
            self.min_value = value;
        }
        if value > self.max_value {
            self.max_value = value;
        }
        true
    }

    /// Record a value with a count.
    pub fn record_n(&mut self, value: u64, count: u64) -> bool {
        if value > self.max_trackable {
            return false;
        }
        let idx = self.counts_index(value);
        if idx >= self.counts.len() {
            return false;
        }
        self.counts[idx] += count;
        self.total_count += count;
        if value < self.min_value {
            self.min_value = value;
        }
        if value > self.max_value {
            self.max_value = value;
        }
        true
    }

    fn counts_index(&self, value: u64) -> usize {
        let bucket = self.bucket_for(value);
        let sub = self.sub_bucket_for(value, bucket);
        self.index_for(bucket, sub)
    }

    fn bucket_for(&self, value: u64) -> u32 {
        let pow2_ceiling = 64 - (value | self.sub_bucket_mask as u64).leading_zeros();
        let sub_bits = (self.sub_bucket_count as f64).log2() as u32 + 1;
        if pow2_ceiling <= sub_bits {
            0
        } else {
            pow2_ceiling - sub_bits
        }
    }

    fn sub_bucket_for(&self, value: u64, bucket: u32) -> u32 {
        (value >> bucket) as u32
    }

    fn index_for(&self, bucket: u32, sub_bucket: u32) -> usize {
        let base = (bucket as usize + 1) * self.sub_bucket_half_count as usize;
        let offset = sub_bucket as usize;
        let half = self.sub_bucket_half_count as usize;
        if offset >= half {
            base + offset - half
        } else {
            offset
        }
    }

    /// Total number of recorded values.
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Minimum recorded value (0 if empty).
    pub fn min(&self) -> u64 {
        if self.total_count == 0 {
            0
        } else {
            self.min_value
        }
    }

    /// Maximum recorded value (0 if empty).
    pub fn max(&self) -> u64 {
        if self.total_count == 0 {
            0
        } else {
            self.max_value
        }
    }

    /// Estimate the value at a given percentile (0.0 - 100.0).
    pub fn percentile(&self, pct: f64) -> u64 {
        if self.total_count == 0 {
            return 0;
        }
        let target = ((pct / 100.0) * self.total_count as f64).ceil() as u64;
        let target = target.max(1);
        let mut cumulative = 0u64;
        for i in 0..self.counts.len() {
            cumulative += self.counts[i];
            if cumulative >= target {
                return self.value_from_index(i);
            }
        }
        self.max_value
    }

    fn value_from_index(&self, index: usize) -> u64 {
        let half = self.sub_bucket_half_count as usize;

        // Indices below half map to bucket 0 with sub_bucket = index
        if index < half {
            return index as u64;
        }

        let mut bucket = 0u32;
        loop {
            let base = (bucket as usize + 1) * half;
            if index < base + half {
                let sub_bucket = (index - base + half) as u64;
                return sub_bucket << bucket;
            }
            bucket += 1;
            if bucket > self.bucket_count {
                break;
            }
        }

        self.max_trackable
    }

    /// Reset the histogram.
    pub fn reset(&mut self) {
        for c in &mut self.counts {
            *c = 0;
        }
        self.total_count = 0;
        self.min_value = u64::MAX;
        self.max_value = 0;
    }

    /// Merge another HDR histogram into this one.
    pub fn merge(&mut self, other: &HdrHistogram) -> bool {
        if self.counts.len() != other.counts.len() {
            return false;
        }
        for (i, c) in other.counts.iter().enumerate() {
            self.counts[i] += c;
        }
        self.total_count += other.total_count;
        if other.total_count > 0 {
            if other.min_value < self.min_value {
                self.min_value = other.min_value;
            }
            if other.max_value > self.max_value {
                self.max_value = other.max_value;
            }
        }
        true
    }
}

// ── Histogram Snapshot ──────────────────────────────────────

/// An immutable snapshot of a bucket histogram's state.
#[derive(Debug, Clone)]
pub struct HistogramSnapshot {
    pub bounds: Vec<f64>,
    pub counts: Vec<u64>,
    pub sum: f64,
    pub count: u64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

impl BucketHistogram {
    /// Take a snapshot of the current state.
    pub fn snapshot(&self) -> HistogramSnapshot {
        HistogramSnapshot {
            bounds: self.bounds.clone(),
            counts: self.counts.clone(),
            sum: self.sum,
            count: self.count,
            min: self.min(),
            max: self.max(),
            mean: self.mean(),
            p50: self.percentile(50.0),
            p90: self.percentile(90.0),
            p95: self.percentile(95.0),
            p99: self.percentile(99.0),
        }
    }
}

impl fmt::Display for BucketHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Histogram(count={}, sum={:.3}, mean={:.3}, min={:.3}, max={:.3})",
            self.count,
            self.sum,
            self.mean(),
            self.min(),
            self.max(),
        )
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_config_explicit() {
        let config = BucketConfig::Explicit(vec![10.0, 5.0, 1.0]);
        let bounds = config.bounds();
        assert_eq!(bounds, vec![1.0, 5.0, 10.0]);
    }

    #[test]
    fn test_bucket_config_linear() {
        let config = BucketConfig::Linear {
            start: 0.0,
            width: 10.0,
            count: 4,
        };
        assert_eq!(config.bounds(), vec![0.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_bucket_config_exponential() {
        let config = BucketConfig::Exponential {
            start: 1.0,
            factor: 2.0,
            count: 4,
        };
        assert_eq!(config.bounds(), vec![1.0, 2.0, 4.0, 8.0]);
    }

    #[test]
    fn test_histogram_new() {
        let h = BucketHistogram::new(&[1.0, 5.0, 10.0]);
        assert_eq!(h.bounds(), &[1.0, 5.0, 10.0]);
        assert_eq!(h.count(), 0);
        assert_eq!(h.sum(), 0.0);
    }

    #[test]
    fn test_histogram_observe() {
        let mut h = BucketHistogram::new(&[1.0, 5.0, 10.0]);
        h.observe(0.5);
        h.observe(3.0);
        h.observe(7.0);
        h.observe(15.0);
        assert_eq!(h.count(), 4);
        assert!((h.sum() - 25.5).abs() < 1e-9);
        assert!((h.min() - 0.5).abs() < 1e-9);
        assert!((h.max() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_bucket_counts() {
        let mut h = BucketHistogram::new(&[1.0, 5.0, 10.0]);
        h.observe(0.5); // bucket 0 (<=1.0)
        h.observe(1.0); // bucket 0 (<=1.0)
        h.observe(3.0); // bucket 1 (<=5.0)
        h.observe(7.0); // bucket 2 (<=10.0)
        h.observe(15.0); // overflow
        let counts = h.bucket_counts();
        assert_eq!(counts[0], 2); // <=1.0
        assert_eq!(counts[1], 1); // <=5.0
        assert_eq!(counts[2], 1); // <=10.0
        assert_eq!(counts[3], 1); // overflow
    }

    #[test]
    fn test_histogram_cumulative() {
        let mut h = BucketHistogram::new(&[1.0, 5.0, 10.0]);
        h.observe(0.5);
        h.observe(3.0);
        h.observe(7.0);
        h.observe(15.0);
        let cum = h.cumulative_counts();
        assert_eq!(cum[0], 1);
        assert_eq!(cum[1], 2);
        assert_eq!(cum[2], 3);
        assert_eq!(cum[3], 4);
    }

    #[test]
    fn test_histogram_mean() {
        let mut h = BucketHistogram::new(&[10.0]);
        h.observe(2.0);
        h.observe(4.0);
        h.observe(6.0);
        assert!((h.mean() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_percentile_empty() {
        let h = BucketHistogram::new(&[1.0, 5.0]);
        assert_eq!(h.percentile(50.0), 0.0);
    }

    #[test]
    fn test_histogram_percentile() {
        let mut h = BucketHistogram::new(&[10.0, 20.0, 30.0]);
        for _ in 0..50 {
            h.observe(5.0);
        }
        for _ in 0..30 {
            h.observe(15.0);
        }
        for _ in 0..20 {
            h.observe(25.0);
        }
        // p50 should be in first bucket (<=10)
        let p50 = h.percentile(50.0);
        assert!(p50 <= 10.0, "p50={}", p50);
        // p99 should be in third bucket (<=30)
        let p99 = h.percentile(99.0);
        assert!(p99 > 10.0, "p99={}", p99);
    }

    #[test]
    fn test_histogram_merge() {
        let mut h1 = BucketHistogram::new(&[1.0, 5.0]);
        h1.observe(0.5);
        h1.observe(3.0);

        let mut h2 = BucketHistogram::new(&[1.0, 5.0]);
        h2.observe(0.8);
        h2.observe(4.0);

        h1.merge(&h2).unwrap();
        assert_eq!(h1.count(), 4);
        assert!((h1.sum() - 8.3).abs() < 1e-9);
        assert!((h1.min() - 0.5).abs() < 1e-9);
        assert!((h1.max() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_merge_different_bounds() {
        let mut h1 = BucketHistogram::new(&[1.0, 5.0]);
        let h2 = BucketHistogram::new(&[1.0, 10.0]);
        assert!(h1.merge(&h2).is_err());
    }

    #[test]
    fn test_histogram_reset() {
        let mut h = BucketHistogram::new(&[10.0]);
        h.observe(5.0);
        h.observe(8.0);
        h.reset();
        assert_eq!(h.count(), 0);
        assert_eq!(h.sum(), 0.0);
        assert_eq!(h.min(), 0.0);
        assert_eq!(h.max(), 0.0);
    }

    #[test]
    fn test_histogram_snapshot() {
        let mut h = BucketHistogram::new(&[10.0, 50.0, 100.0]);
        for v in [5.0, 15.0, 25.0, 75.0, 95.0] {
            h.observe(v);
        }
        let snap = h.snapshot();
        assert_eq!(snap.count, 5);
        assert!((snap.sum - 215.0).abs() < 1e-9);
        assert!((snap.min - 5.0).abs() < 1e-9);
        assert!((snap.max - 95.0).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_from_config() {
        let config = BucketConfig::Linear {
            start: 0.0,
            width: 5.0,
            count: 3,
        };
        let h = BucketHistogram::from_config(&config);
        assert_eq!(h.bounds(), &[0.0, 5.0, 10.0]);
    }

    #[test]
    fn test_histogram_display() {
        let mut h = BucketHistogram::new(&[10.0]);
        h.observe(5.0);
        let text = format!("{}", h);
        assert!(text.contains("count=1"));
        assert!(text.contains("sum=5.000"));
    }

    #[test]
    fn test_hdr_histogram_basic() {
        let mut h = HdrHistogram::new(3_600_000, 3);
        assert!(h.record(100));
        assert!(h.record(200));
        assert!(h.record(300));
        assert_eq!(h.total_count(), 3);
        assert!(h.min() <= 100);
        assert!(h.max() >= 300);
    }

    #[test]
    fn test_hdr_histogram_out_of_range() {
        let mut h = HdrHistogram::new(1000, 2);
        assert!(!h.record(2000));
        assert_eq!(h.total_count(), 0);
    }

    #[test]
    fn test_hdr_histogram_record_n() {
        let mut h = HdrHistogram::new(1000, 2);
        h.record_n(50, 10);
        assert_eq!(h.total_count(), 10);
    }

    #[test]
    fn test_hdr_histogram_percentile() {
        let mut h = HdrHistogram::new(10000, 3);
        for v in 1..=100 {
            h.record(v);
        }
        let p50 = h.percentile(50.0);
        // Should be approximately 50, within HDR precision
        assert!(p50 >= 40 && p50 <= 60, "p50={}", p50);
    }

    #[test]
    fn test_hdr_histogram_reset() {
        let mut h = HdrHistogram::new(1000, 2);
        h.record(100);
        h.record(200);
        h.reset();
        assert_eq!(h.total_count(), 0);
        assert_eq!(h.min(), 0);
        assert_eq!(h.max(), 0);
    }

    #[test]
    fn test_hdr_histogram_merge() {
        let mut h1 = HdrHistogram::new(1000, 2);
        h1.record(100);
        let mut h2 = HdrHistogram::new(1000, 2);
        h2.record(200);
        assert!(h1.merge(&h2));
        assert_eq!(h1.total_count(), 2);
        assert!(h1.max() >= 200);
    }

    #[test]
    fn test_hdr_histogram_empty_percentile() {
        let h = HdrHistogram::new(1000, 2);
        assert_eq!(h.percentile(50.0), 0);
        assert_eq!(h.percentile(99.0), 0);
    }

    #[test]
    fn test_min_max_empty() {
        let h = BucketHistogram::new(&[10.0]);
        assert_eq!(h.min(), 0.0);
        assert_eq!(h.max(), 0.0);
    }
}
