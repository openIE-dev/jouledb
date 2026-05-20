//! Time series data structure — append-only with downsampling, range queries, and analysis.
//!
//! Replaces timeseries libraries with a pure-Rust time series engine.
//! Supports append-only insertion, downsampling (avg/min/max/last), range queries,
//! gap detection, linear interpolation, moving average, and series merge.

use serde::{Deserialize, Serialize};

// ── Data Point ────────────────────────────────────────────────

/// A single time series data point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp in milliseconds.
    pub timestamp: u64,
    /// Value at this timestamp.
    pub value: f64,
}

impl DataPoint {
    /// Create a new data point.
    pub fn new(timestamp: u64, value: f64) -> Self {
        Self { timestamp, value }
    }
}

// ── Downsample Method ─────────────────────────────────────────

/// Method for aggregating data points within a bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownsampleMethod {
    /// Average of values in the bucket.
    Avg,
    /// Minimum value in the bucket.
    Min,
    /// Maximum value in the bucket.
    Max,
    /// Last value in the bucket.
    Last,
    /// First value in the bucket.
    First,
    /// Sum of values in the bucket.
    Sum,
    /// Count of values in the bucket.
    Count,
}

// ── Gap ───────────────────────────────────────────────────────

/// A detected gap in the time series.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gap {
    /// Timestamp of the point before the gap.
    pub before: u64,
    /// Timestamp of the point after the gap.
    pub after: u64,
    /// Duration of the gap in ms.
    pub duration_ms: u64,
}

// ── Time Series Stats ─────────────────────────────────────────

/// Statistics about a time series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesStats {
    pub count: usize,
    pub min_value: f64,
    pub max_value: f64,
    pub avg_value: f64,
    pub sum_value: f64,
    pub first_timestamp: u64,
    pub last_timestamp: u64,
    pub time_span_ms: u64,
}

// ── Time Series ───────────────────────────────────────────────

/// Append-only time series data structure.
#[derive(Debug, Clone)]
pub struct TimeSeries {
    /// Label for this series.
    label: String,
    /// Data points, sorted by timestamp.
    points: Vec<DataPoint>,
}

impl TimeSeries {
    /// Create a new empty time series.
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            points: Vec::new(),
        }
    }

    /// Create a time series from existing data points (sorts them).
    pub fn from_points(label: &str, mut points: Vec<DataPoint>) -> Self {
        points.sort_by_key(|p| p.timestamp);
        Self {
            label: label.to_string(),
            points,
        }
    }

    /// Label of the series.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Append a data point. Must have timestamp >= last point.
    /// Returns false if timestamp is before the last point.
    pub fn append(&mut self, timestamp: u64, value: f64) -> bool {
        if let Some(last) = self.points.last() {
            if timestamp < last.timestamp {
                return false;
            }
        }
        self.points.push(DataPoint::new(timestamp, value));
        true
    }

    /// Number of data points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Is the series empty?
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get all data points.
    pub fn points(&self) -> &[DataPoint] {
        &self.points
    }

    /// Get a point by index.
    pub fn get(&self, index: usize) -> Option<&DataPoint> {
        self.points.get(index)
    }

    /// First data point.
    pub fn first(&self) -> Option<&DataPoint> {
        self.points.first()
    }

    /// Last data point.
    pub fn last(&self) -> Option<&DataPoint> {
        self.points.last()
    }

    /// Range query: all points with timestamp in [start, end).
    pub fn range(&self, start: u64, end: u64) -> Vec<DataPoint> {
        self.points
            .iter()
            .filter(|p| p.timestamp >= start && p.timestamp < end)
            .copied()
            .collect()
    }

    /// Downsample the series into buckets of `bucket_ms` using the given method.
    pub fn downsample(
        &self,
        bucket_ms: u64,
        method: DownsampleMethod,
    ) -> Vec<DataPoint> {
        if self.points.is_empty() || bucket_ms == 0 {
            return Vec::new();
        }

        let first_ts = self.points[0].timestamp;
        let last_ts = self.points[self.points.len() - 1].timestamp;

        let mut result = Vec::new();
        let mut bucket_start = first_ts;

        while bucket_start <= last_ts {
            let bucket_end = bucket_start.saturating_add(bucket_ms);
            let bucket_points: Vec<f64> = self
                .points
                .iter()
                .filter(|p| p.timestamp >= bucket_start && p.timestamp < bucket_end)
                .map(|p| p.value)
                .collect();

            if !bucket_points.is_empty() {
                let value = aggregate(&bucket_points, method);
                result.push(DataPoint::new(bucket_start, value));
            }

            bucket_start = bucket_end;
        }

        result
    }

    /// Detect gaps larger than `threshold_ms` between consecutive points.
    pub fn detect_gaps(&self, threshold_ms: u64) -> Vec<Gap> {
        let mut gaps = Vec::new();
        for window in self.points.windows(2) {
            let dt = window[1].timestamp.saturating_sub(window[0].timestamp);
            if dt > threshold_ms {
                gaps.push(Gap {
                    before: window[0].timestamp,
                    after: window[1].timestamp,
                    duration_ms: dt,
                });
            }
        }
        gaps
    }

    /// Linear interpolation at the given timestamp.
    /// Returns None if timestamp is outside the series range.
    pub fn interpolate(&self, timestamp: u64) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }

        // Exact match.
        if let Some(p) = self.points.iter().find(|p| p.timestamp == timestamp) {
            return Some(p.value);
        }

        // Find surrounding points.
        let right_idx = self.points.iter().position(|p| p.timestamp > timestamp)?;
        if right_idx == 0 {
            return None; // Before first point.
        }

        let left = &self.points[right_idx - 1];
        let right = &self.points[right_idx];

        let dt_total = (right.timestamp - left.timestamp) as f64;
        if dt_total == 0.0 {
            return Some(left.value);
        }

        let dt_point = (timestamp - left.timestamp) as f64;
        let ratio = dt_point / dt_total;
        let interpolated = left.value + ratio * (right.value - left.value);
        Some(interpolated)
    }

    /// Compute a simple moving average with the given window size (in number of points).
    pub fn moving_average(&self, window_size: usize) -> Vec<DataPoint> {
        if window_size == 0 || self.points.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();

        for i in 0..self.points.len() {
            let start = if i + 1 >= window_size {
                i + 1 - window_size
            } else {
                0
            };
            let slice = &self.points[start..=i];
            let avg: f64 = slice.iter().map(|p| p.value).sum::<f64>() / slice.len() as f64;
            result.push(DataPoint::new(self.points[i].timestamp, avg));
        }

        result
    }

    /// Compute statistics for the series.
    pub fn stats(&self) -> Option<TimeSeriesStats> {
        if self.points.is_empty() {
            return None;
        }

        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;
        let mut sum = 0.0;

        for p in &self.points {
            if p.value < min_val {
                min_val = p.value;
            }
            if p.value > max_val {
                max_val = p.value;
            }
            sum += p.value;
        }

        let first_ts = self.points[0].timestamp;
        let last_ts = self.points[self.points.len() - 1].timestamp;

        Some(TimeSeriesStats {
            count: self.points.len(),
            min_value: min_val,
            max_value: max_val,
            avg_value: sum / self.points.len() as f64,
            sum_value: sum,
            first_timestamp: first_ts,
            last_timestamp: last_ts,
            time_span_ms: last_ts - first_ts,
        })
    }

    /// Merge two time series into one, interleaving by timestamp.
    /// When timestamps match, values from `other` take precedence.
    pub fn merge(&self, other: &TimeSeries) -> TimeSeries {
        let mut merged = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.points.len() && j < other.points.len() {
            if self.points[i].timestamp < other.points[j].timestamp {
                merged.push(self.points[i]);
                i += 1;
            } else if self.points[i].timestamp > other.points[j].timestamp {
                merged.push(other.points[j]);
                j += 1;
            } else {
                // Same timestamp — other takes precedence.
                merged.push(other.points[j]);
                i += 1;
                j += 1;
            }
        }

        while i < self.points.len() {
            merged.push(self.points[i]);
            i += 1;
        }
        while j < other.points.len() {
            merged.push(other.points[j]);
            j += 1;
        }

        let label = format!("{}+{}", self.label, other.label);
        TimeSeries {
            label,
            points: merged,
        }
    }

    /// Compute the rate of change (derivative) between consecutive points.
    /// Returns values in units per millisecond.
    pub fn rate_of_change(&self) -> Vec<DataPoint> {
        let mut result = Vec::new();
        for window in self.points.windows(2) {
            let dt = (window[1].timestamp - window[0].timestamp) as f64;
            if dt > 0.0 {
                let dv = window[1].value - window[0].value;
                let rate = dv / dt;
                result.push(DataPoint::new(window[1].timestamp, rate));
            }
        }
        result
    }
}

/// Aggregate a non-empty slice of values using the given method.
fn aggregate(values: &[f64], method: DownsampleMethod) -> f64 {
    match method {
        DownsampleMethod::Avg => values.iter().sum::<f64>() / values.len() as f64,
        DownsampleMethod::Min => values.iter().cloned().fold(f64::MAX, f64::min),
        DownsampleMethod::Max => values.iter().cloned().fold(f64::MIN, f64::max),
        DownsampleMethod::Last => *values.last().unwrap(),
        DownsampleMethod::First => *values.first().unwrap(),
        DownsampleMethod::Sum => values.iter().sum(),
        DownsampleMethod::Count => values.len() as f64,
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_series() -> TimeSeries {
        let mut ts = TimeSeries::new("temp");
        ts.append(1000, 20.0);
        ts.append(2000, 22.0);
        ts.append(3000, 21.0);
        ts.append(4000, 25.0);
        ts.append(5000, 23.0);
        ts
    }

    #[test]
    fn test_append_and_len() {
        let ts = sample_series();
        assert_eq!(ts.len(), 5);
        assert!(!ts.is_empty());
    }

    #[test]
    fn test_append_out_of_order_rejected() {
        let mut ts = TimeSeries::new("test");
        ts.append(100, 1.0);
        assert!(!ts.append(50, 2.0)); // Before last — rejected.
        assert_eq!(ts.len(), 1);
    }

    #[test]
    fn test_first_last() {
        let ts = sample_series();
        assert_eq!(ts.first().unwrap().timestamp, 1000);
        assert_eq!(ts.last().unwrap().timestamp, 5000);
    }

    #[test]
    fn test_range_query() {
        let ts = sample_series();
        let range = ts.range(2000, 4000);
        assert_eq!(range.len(), 2);
        assert_eq!(range[0].value, 22.0);
        assert_eq!(range[1].value, 21.0);
    }

    #[test]
    fn test_downsample_avg() {
        let ts = sample_series();
        let ds = ts.downsample(2000, DownsampleMethod::Avg);
        assert_eq!(ds.len(), 3);
        // [1000,3000): 20, 22 -> avg 21
        assert!((ds[0].value - 21.0).abs() < 0.01);
    }

    #[test]
    fn test_downsample_min() {
        let ts = sample_series();
        let ds = ts.downsample(2000, DownsampleMethod::Min);
        // [1000,3000): min(20, 22) = 20
        assert_eq!(ds[0].value, 20.0);
    }

    #[test]
    fn test_downsample_max() {
        let ts = sample_series();
        let ds = ts.downsample(2000, DownsampleMethod::Max);
        // [1000,3000): max(20, 22) = 22
        assert_eq!(ds[0].value, 22.0);
    }

    #[test]
    fn test_downsample_last() {
        let ts = sample_series();
        let ds = ts.downsample(2000, DownsampleMethod::Last);
        // [1000,3000): last = 22
        assert_eq!(ds[0].value, 22.0);
    }

    #[test]
    fn test_downsample_empty() {
        let ts = TimeSeries::new("empty");
        let ds = ts.downsample(1000, DownsampleMethod::Avg);
        assert!(ds.is_empty());
    }

    #[test]
    fn test_gap_detection() {
        let mut ts = TimeSeries::new("gaps");
        ts.append(1000, 1.0);
        ts.append(2000, 2.0);
        ts.append(10000, 3.0); // 8-second gap.
        ts.append(11000, 4.0);
        let gaps = ts.detect_gaps(5000);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].before, 2000);
        assert_eq!(gaps[0].after, 10000);
        assert_eq!(gaps[0].duration_ms, 8000);
    }

    #[test]
    fn test_no_gaps() {
        let ts = sample_series();
        let gaps = ts.detect_gaps(5000);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_interpolation_exact() {
        let ts = sample_series();
        let val = ts.interpolate(2000).unwrap();
        assert_eq!(val, 22.0);
    }

    #[test]
    fn test_interpolation_between() {
        let ts = sample_series();
        // Between 1000 (20.0) and 2000 (22.0), at 1500:
        let val = ts.interpolate(1500).unwrap();
        assert!((val - 21.0).abs() < 0.01);
    }

    #[test]
    fn test_interpolation_outside() {
        let ts = sample_series();
        assert!(ts.interpolate(500).is_none());
    }

    #[test]
    fn test_moving_average() {
        let ts = sample_series();
        let ma = ts.moving_average(3);
        assert_eq!(ma.len(), 5);
        // Point 2: avg(20, 22, 21) = 21.0
        assert!((ma[2].value - 21.0).abs() < 0.01);
    }

    #[test]
    fn test_moving_average_window_1() {
        let ts = sample_series();
        let ma = ts.moving_average(1);
        // Window of 1 = original values.
        for (orig, avg) in ts.points().iter().zip(ma.iter()) {
            assert!((orig.value - avg.value).abs() < 0.001);
        }
    }

    #[test]
    fn test_stats() {
        let ts = sample_series();
        let s = ts.stats().unwrap();
        assert_eq!(s.count, 5);
        assert_eq!(s.min_value, 20.0);
        assert_eq!(s.max_value, 25.0);
        assert!((s.avg_value - 22.2).abs() < 0.01);
        assert_eq!(s.time_span_ms, 4000);
    }

    #[test]
    fn test_stats_empty() {
        let ts = TimeSeries::new("empty");
        assert!(ts.stats().is_none());
    }

    #[test]
    fn test_merge() {
        let mut a = TimeSeries::new("a");
        a.append(1000, 1.0);
        a.append(3000, 3.0);

        let mut b = TimeSeries::new("b");
        b.append(2000, 2.0);
        b.append(4000, 4.0);

        let merged = a.merge(&b);
        assert_eq!(merged.len(), 4);
        assert_eq!(merged.points()[0].timestamp, 1000);
        assert_eq!(merged.points()[1].timestamp, 2000);
        assert_eq!(merged.points()[2].timestamp, 3000);
        assert_eq!(merged.points()[3].timestamp, 4000);
    }

    #[test]
    fn test_merge_overlapping() {
        let mut a = TimeSeries::new("a");
        a.append(1000, 10.0);

        let mut b = TimeSeries::new("b");
        b.append(1000, 20.0); // Same timestamp — b takes precedence.

        let merged = a.merge(&b);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged.points()[0].value, 20.0);
    }

    #[test]
    fn test_from_points_sorts() {
        let points = vec![
            DataPoint::new(3000, 3.0),
            DataPoint::new(1000, 1.0),
            DataPoint::new(2000, 2.0),
        ];
        let ts = TimeSeries::from_points("sorted", points);
        assert_eq!(ts.points()[0].timestamp, 1000);
        assert_eq!(ts.points()[2].timestamp, 3000);
    }

    #[test]
    fn test_rate_of_change() {
        let mut ts = TimeSeries::new("roc");
        ts.append(0, 0.0);
        ts.append(1000, 10.0); // 10/1000 = 0.01 per ms
        ts.append(2000, 30.0); // 20/1000 = 0.02 per ms
        let roc = ts.rate_of_change();
        assert_eq!(roc.len(), 2);
        assert!((roc[0].value - 0.01).abs() < 0.0001);
        assert!((roc[1].value - 0.02).abs() < 0.0001);
    }

    #[test]
    fn test_label() {
        let ts = TimeSeries::new("my-series");
        assert_eq!(ts.label(), "my-series");
    }

    #[test]
    fn test_get_by_index() {
        let ts = sample_series();
        assert_eq!(ts.get(0).unwrap().value, 20.0);
        assert!(ts.get(100).is_none());
    }
}
