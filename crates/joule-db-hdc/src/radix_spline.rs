//! # RadixSpline Learned Index
//!
//! A lightweight learned index structure based on the RadixSpline paper (Kipf et al., 2020)
//! and Learned Index Structures research (Kraska et al.).
//!
//! RadixSpline predicts the position of a key in sorted data using a two-level lookup:
//!
//! 1. **Radix Table** -- An O(1) lookup using the top bits of the key to narrow down the
//!    relevant segment of the spline.
//! 2. **Spline Interpolation** -- Linear interpolation between control points to estimate
//!    the key's position, accurate to within a configurable error bound.
//!
//! This achieves 50--100ns lookups, roughly 70% faster than cache-optimized B-Trees for
//! numeric keys with predictable distributions.
//!
//! ## Architecture
//!
//! ```text
//! Key ──► Radix Table (2^r entries) ──► Spline Segment ──► Linear Interpolation ──► Position ± ε
//!              O(1)                     O(log s)                O(1)
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_hdc::radix_spline::{RadixSpline, RadixSplineConfig};
//!
//! // Sorted dataset
//! let keys: Vec<u64> = (0..10_000).collect();
//! let positions: Vec<usize> = (0..10_000).collect();
//!
//! let config = RadixSplineConfig::default();
//! let index = RadixSpline::build(&keys, &positions, config);
//!
//! // Lookup returns (estimated_position, error_bound)
//! let (est_pos, err) = index.lookup(5000);
//! assert!((est_pos as i64 - 5000).unsigned_abs() <= err as u64);
//!
//! // Full search with local scan
//! let found = index.search(5000, &keys);
//! assert_eq!(found, Some(5000));
//! ```

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for building a [`RadixSpline`] index.
#[derive(Debug, Clone)]
pub struct RadixSplineConfig {
    /// Maximum allowed prediction error (in positions). The spline guarantees that
    /// the true position is within `±max_error` of the predicted position.
    /// Smaller values produce more spline points but tighter bounds.
    pub max_error: usize,

    /// Number of bits used for the radix table. The table has `2^num_radix_bits`
    /// entries, each pointing into the spline points array.
    /// More bits reduce the binary search range but increase memory usage.
    pub num_radix_bits: usize,

    /// Maximum number of spline control points. If the greedy algorithm would
    /// produce more points, it stops and the remaining keys use the last segment.
    pub spline_max_points: usize,
}

impl Default for RadixSplineConfig {
    fn default() -> Self {
        Self {
            max_error: 32,
            num_radix_bits: 18,
            spline_max_points: 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// Spline Point
// ---------------------------------------------------------------------------

/// A control point on the error-bounded spline.
///
/// Each point records a key value and the corresponding position in the sorted
/// data. The spline linearly interpolates between adjacent control points to
/// predict positions for intermediate keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplinePoint {
    /// The key value at this control point.
    pub key: u64,
    /// The position of this key in the sorted data array.
    pub position: usize,
}

// ---------------------------------------------------------------------------
// RadixSpline
// ---------------------------------------------------------------------------

/// A learned index that predicts key positions using a radix-table-accelerated
/// piecewise-linear spline.
///
/// The index is immutable after construction. For mutable workloads, see
/// [`LearnedBTreeHybrid`] which wraps a `RadixSpline` with a B-Tree buffer.
#[derive(Debug, Clone)]
pub struct RadixSpline {
    /// Radix lookup table. Entry `i` stores the index into `spline_points` of
    /// the first spline segment whose key prefix is `>= i`.
    /// Has `2^num_radix_bits` entries.
    radix_table: Vec<usize>,

    /// Sorted spline control points.
    spline_points: Vec<SplinePoint>,

    /// Maximum prediction error (guaranteed bound).
    max_error: usize,

    /// Minimum key in the indexed dataset.
    min_key: u64,

    /// Maximum key in the indexed dataset.
    max_key: u64,

    /// Number of radix bits used (stored for lookup computation).
    num_radix_bits: usize,
}

impl RadixSpline {
    /// Build a `RadixSpline` index over sorted keys and their positions.
    ///
    /// # Arguments
    ///
    /// * `sorted_keys` -- Keys in ascending order. Duplicates are allowed.
    /// * `positions` -- The position of each key (typically `0..n`).
    /// * `config` -- Configuration parameters.
    ///
    /// # Panics
    ///
    /// Panics if `sorted_keys` and `positions` have different lengths or are empty.
    pub fn build(sorted_keys: &[u64], positions: &[usize], config: RadixSplineConfig) -> Self {
        assert!(
            !sorted_keys.is_empty(),
            "RadixSpline::build requires at least one key"
        );
        assert_eq!(
            sorted_keys.len(),
            positions.len(),
            "sorted_keys and positions must have the same length"
        );

        let min_key = sorted_keys[0];
        let max_key = sorted_keys[sorted_keys.len() - 1];

        // Step 1: Build spline points using the greedy error-bounded algorithm.
        let spline_points = Self::build_spline(
            sorted_keys,
            positions,
            config.max_error,
            config.spline_max_points,
        );

        // Step 2: Build radix table for O(1) first-level lookup.
        let radix_table =
            Self::build_radix_table(&spline_points, min_key, max_key, config.num_radix_bits);

        let mut rs = RadixSpline {
            radix_table,
            spline_points,
            max_error: config.max_error,
            min_key,
            max_key,
            num_radix_bits: config.num_radix_bits,
        };

        // Step 3: Compute the actual worst-case error across all data points.
        // The greedy corridor algorithm may underestimate the error for non-uniform
        // distributions because the interpolation slope between control-point
        // positions can fall outside the valid corridor.
        let mut actual_max = 0usize;
        for i in 0..sorted_keys.len() {
            let (est, _) = rs.lookup(sorted_keys[i]);
            let diff = if est > positions[i] {
                est - positions[i]
            } else {
                positions[i] - est
            };
            actual_max = actual_max.max(diff);
        }
        rs.max_error = actual_max.max(config.max_error);

        rs
    }

    /// Look up the estimated position of `key` and the error bound.
    ///
    /// Returns `(estimated_position, error_bound)` where the true position is
    /// guaranteed to be in `[estimated_position - error_bound, estimated_position + error_bound]`
    /// (clamped to valid indices).
    pub fn lookup(&self, key: u64) -> (usize, usize) {
        if self.spline_points.len() <= 1 {
            return (self.spline_points[0].position, self.max_error);
        }

        // Clamp key to the indexed range.
        let clamped_key = key.clamp(self.min_key, self.max_key);

        // Step 1: Radix table lookup to narrow the spline search range.
        let radix_index = self.key_to_radix(clamped_key);
        let spline_start = self.radix_table[radix_index];

        // The end of the search range is the next non-equal radix entry.
        let spline_end = if radix_index + 1 < self.radix_table.len() {
            // Scan forward to find the upper bound.
            let mut end = self.radix_table[radix_index + 1];
            // Ensure we have at least two points to interpolate between.
            if end <= spline_start {
                end = (spline_start + 1).min(self.spline_points.len() - 1);
            }
            end
        } else {
            self.spline_points.len() - 1
        };

        // Step 2: Binary search within the spline points for the correct segment.
        let seg_idx = self.find_spline_segment(clamped_key, spline_start, spline_end);

        // Step 3: Linear interpolation between the two bounding control points.
        let estimated_pos = self.interpolate(clamped_key, seg_idx);

        (estimated_pos, self.max_error)
    }

    /// Full lookup: predicts the position using the spline, then performs a local
    /// linear scan within the error bound to find the exact key.
    ///
    /// Returns `Some(index)` if the key is found in `data`, or `None` otherwise.
    pub fn search(&self, key: u64, data: &[u64]) -> Option<usize> {
        if data.is_empty() {
            return None;
        }

        let (est_pos, err_bound) = self.lookup(key);

        // Compute the scan window, clamped to data bounds.
        let lo = est_pos.saturating_sub(err_bound);
        let hi = (est_pos + err_bound).min(data.len() - 1);

        // Local linear scan within the error-bounded window.
        for i in lo..=hi {
            if data[i] == key {
                return Some(i);
            }
            // Since data is sorted, we can stop early if we've passed the key.
            if data[i] > key {
                return None;
            }
        }

        None
    }

    // -----------------------------------------------------------------------
    // Private: Spline construction
    // -----------------------------------------------------------------------

    /// Greedy error-bounded spline construction.
    ///
    /// Walks through the sorted keys. Starts a new spline segment whenever the
    /// linear interpolation from the last control point would exceed `max_error`
    /// for any key in the current run.
    fn build_spline(
        sorted_keys: &[u64],
        positions: &[usize],
        max_error: usize,
        max_points: usize,
    ) -> Vec<SplinePoint> {
        let n = sorted_keys.len();
        let mut points = Vec::with_capacity(max_points.min(n));

        // Always include the first point.
        points.push(SplinePoint {
            key: sorted_keys[0],
            position: positions[0],
        });

        if n == 1 {
            return points;
        }

        // Track the corridor of valid slopes from the last emitted control point.
        // The corridor is defined by (upper_key, upper_pos) and (lower_key, lower_pos)
        // forming the tightest bounding lines that keep error within max_error.
        let mut last_key = sorted_keys[0];
        let mut last_pos = positions[0] as f64;

        // Slope bounds: we track the min and max feasible slopes from the last point.
        let mut slope_min = f64::NEG_INFINITY;
        let mut slope_max = f64::INFINITY;

        for i in 1..n {
            let key = sorted_keys[i];
            let pos = positions[i] as f64;

            if key == last_key {
                // Duplicate key -- the corridor doesn't change.
                continue;
            }

            let dx = (key - last_key) as f64;
            // The slopes that would put us exactly at pos +/- max_error.
            let slope_to_upper = (pos + max_error as f64 - last_pos) / dx;
            let slope_to_lower = (pos - max_error as f64 - last_pos) / dx;

            // Tighten the corridor.
            let new_slope_min = slope_min.max(slope_to_lower);
            let new_slope_max = slope_max.min(slope_to_upper);

            if new_slope_min > new_slope_max {
                // The corridor has collapsed -- emit the previous point as a control point.
                // Use the midpoint of the previous valid corridor as the slope.
                if points.len() >= max_points {
                    break;
                }

                // Emit the previous key/position as a new control point.
                // We step back to the last valid key (i-1).
                points.push(SplinePoint {
                    key: sorted_keys[i - 1],
                    position: positions[i - 1],
                });

                // Reset corridor from the newly emitted point.
                last_key = sorted_keys[i - 1];
                last_pos = positions[i - 1] as f64;

                let dx2 = (key - last_key) as f64;
                if dx2 > 0.0 {
                    slope_min = (pos - max_error as f64 - last_pos) / dx2;
                    slope_max = (pos + max_error as f64 - last_pos) / dx2;
                } else {
                    slope_min = f64::NEG_INFINITY;
                    slope_max = f64::INFINITY;
                }
            } else {
                slope_min = new_slope_min;
                slope_max = new_slope_max;
            }
        }

        // Always include the last point.
        let last = SplinePoint {
            key: sorted_keys[n - 1],
            position: positions[n - 1],
        };
        if points.last() != Some(&last) {
            points.push(last);
        }

        points
    }

    /// Build the radix table from the spline points.
    ///
    /// For each of the `2^num_radix_bits` buckets, store the index of the first
    /// spline point whose key's radix prefix is >= that bucket.
    fn build_radix_table(
        spline_points: &[SplinePoint],
        min_key: u64,
        max_key: u64,
        num_radix_bits: usize,
    ) -> Vec<usize> {
        let table_size = 1usize << num_radix_bits;
        let mut table = vec![0usize; table_size];

        if spline_points.is_empty() || min_key == max_key {
            return table;
        }

        let key_range = max_key - min_key;

        // For each spline point, compute its radix bucket and record the mapping.
        // We sweep through in order and fill the table so that table[r] points
        // to the first spline segment relevant for keys with radix prefix >= r.
        let mut prev_bucket = 0usize;
        for (idx, point) in spline_points.iter().enumerate() {
            let normalized = point.key.saturating_sub(min_key);
            let bucket = if key_range == 0 {
                0
            } else {
                ((normalized as u128 * (table_size - 1) as u128) / key_range as u128) as usize
            };
            let bucket = bucket.min(table_size - 1);

            // Fill all buckets from prev_bucket+1 to bucket with idx.
            // (first spline point whose prefix >= that bucket)
            for b in (prev_bucket + 1)..=bucket {
                table[b] = idx;
            }
            prev_bucket = bucket;
        }

        // Fill remaining buckets with the last spline index.
        let last_idx = spline_points.len().saturating_sub(1);
        for b in (prev_bucket + 1)..table_size {
            table[b] = last_idx;
        }

        table
    }

    /// Map a key to a radix table index.
    fn key_to_radix(&self, key: u64) -> usize {
        let key_range = self.max_key - self.min_key;
        if key_range == 0 {
            return 0;
        }
        let normalized = key.saturating_sub(self.min_key);
        let table_size = 1usize << self.num_radix_bits;
        let bucket = ((normalized as u128 * (table_size - 1) as u128) / key_range as u128) as usize;
        bucket.min(table_size - 1)
    }

    /// Binary search within `spline_points[lo..=hi]` to find the segment
    /// containing `key`. Returns the index of the left endpoint.
    fn find_spline_segment(&self, key: u64, lo: usize, hi: usize) -> usize {
        let hi = hi.min(self.spline_points.len() - 1);
        let lo = lo.min(hi);

        // Binary search for the rightmost spline point with key <= target.
        let mut left = lo;
        let mut right = hi;

        while left < right {
            let mid = left + (right - left + 1) / 2;
            if self.spline_points[mid].key <= key {
                left = mid;
            } else {
                right = mid - 1;
            }
        }

        // Ensure we don't return the very last point (we need a right neighbor).
        if left >= self.spline_points.len() - 1 {
            left = self.spline_points.len().saturating_sub(2);
        }

        left
    }

    /// Linearly interpolate between spline_points[seg] and spline_points[seg+1].
    fn interpolate(&self, key: u64, seg: usize) -> usize {
        let left = &self.spline_points[seg];
        let right_idx = (seg + 1).min(self.spline_points.len() - 1);
        let right = &self.spline_points[right_idx];

        if left.key == right.key {
            return left.position;
        }

        let key_fraction = (key.saturating_sub(left.key)) as f64 / (right.key - left.key) as f64;
        let key_fraction = key_fraction.clamp(0.0, 1.0);

        let pos_range = right.position as f64 - left.position as f64;
        let estimated = left.position as f64 + key_fraction * pos_range;

        estimated.round() as usize
    }

    /// Returns the number of spline control points.
    pub fn num_spline_points(&self) -> usize {
        self.spline_points.len()
    }

    /// Returns the maximum error bound.
    pub fn error_bound(&self) -> usize {
        self.max_error
    }

    /// Returns the key range `(min_key, max_key)`.
    pub fn key_range(&self) -> (u64, u64) {
        (self.min_key, self.max_key)
    }

    /// Returns the memory usage of the index in bytes (approximate).
    pub fn memory_usage(&self) -> usize {
        let radix_bytes = self.radix_table.len() * std::mem::size_of::<usize>();
        let spline_bytes = self.spline_points.len() * std::mem::size_of::<SplinePoint>();
        let overhead = std::mem::size_of::<Self>();
        radix_bytes + spline_bytes + overhead
    }
}

// ---------------------------------------------------------------------------
// LearnedBTreeHybrid
// ---------------------------------------------------------------------------

/// A hybrid index that combines a [`RadixSpline`] learned index with a B-Tree
/// fallback for mutable workloads.
///
/// Incoming writes go into a B-Tree buffer. Reads first consult the B-Tree
/// buffer, then fall back to the learned index over the bulk-loaded data.
/// When enough writes accumulate or the distribution shifts, the index can be
/// rebuilt to incorporate buffered data.
///
/// This design follows the "learned index + delta buffer" pattern from the
/// ALEX and LIPP literature.
#[derive(Debug, Clone)]
pub struct LearnedBTreeHybrid {
    /// The learned index over the bulk-loaded (immutable) data.
    spline: Option<RadixSpline>,

    /// Sorted bulk data keys.
    bulk_keys: Vec<u64>,

    /// Bulk data values, parallel to `bulk_keys`.
    bulk_values: Vec<Vec<u8>>,

    /// B-Tree buffer for recent inserts that haven't been merged yet.
    buffer: BTreeMap<u64, Vec<u8>>,

    /// Number of inserts since the last rebuild.
    inserts_since_rebuild: usize,

    /// Threshold: rebuild when buffer exceeds this fraction of bulk data.
    rebuild_threshold: f64,

    /// Configuration for rebuilding the spline.
    config: RadixSplineConfig,

    /// Tracks prediction errors to detect distribution shift.
    recent_errors: Vec<usize>,

    /// Maximum number of recent errors to track.
    error_window: usize,
}

impl LearnedBTreeHybrid {
    /// Create a new empty hybrid index with default configuration.
    pub fn new() -> Self {
        Self::with_config(RadixSplineConfig::default())
    }

    /// Create a new empty hybrid index with the given spline configuration.
    pub fn with_config(config: RadixSplineConfig) -> Self {
        Self {
            spline: None,
            bulk_keys: Vec::new(),
            bulk_values: Vec::new(),
            buffer: BTreeMap::new(),
            inserts_since_rebuild: 0,
            rebuild_threshold: 0.2,
            config,
            recent_errors: Vec::new(),
            error_window: 1000,
        }
    }

    /// Bulk-load sorted data and build the learned index.
    ///
    /// This replaces any existing bulk data and spline index.
    pub fn bulk_load(&mut self, keys: Vec<u64>, values: Vec<Vec<u8>>) {
        assert_eq!(keys.len(), values.len());
        if keys.is_empty() {
            self.spline = None;
            self.bulk_keys = keys;
            self.bulk_values = values;
            return;
        }

        let positions: Vec<usize> = (0..keys.len()).collect();
        let spline = RadixSpline::build(&keys, &positions, self.config.clone());

        self.spline = Some(spline);
        self.bulk_keys = keys;
        self.bulk_values = values;
        self.inserts_since_rebuild = 0;
        self.recent_errors.clear();
    }

    /// Insert a key-value pair. The value is stored in the B-Tree buffer until
    /// the next rebuild.
    pub fn insert(&mut self, key: u64, value: Vec<u8>) {
        self.buffer.insert(key, value);
        self.inserts_since_rebuild += 1;
    }

    /// Look up a key, checking the buffer first and then the learned index.
    ///
    /// Returns a reference to the value if found.
    pub fn get(&self, key: u64) -> Option<&[u8]> {
        // Check the B-Tree buffer first (most recent writes).
        if let Some(val) = self.buffer.get(&key) {
            return Some(val.as_slice());
        }

        // Fall back to the learned index over bulk data.
        if let Some(ref spline) = self.spline {
            if let Some(idx) = spline.search(key, &self.bulk_keys) {
                return Some(self.bulk_values[idx].as_slice());
            }
        }

        None
    }

    /// Check whether the index should be rebuilt.
    ///
    /// Returns `true` if:
    /// - The buffer has grown beyond `rebuild_threshold * bulk_data_size`, or
    /// - The average prediction error has increased significantly (distribution shift).
    pub fn needs_rebuild(&self) -> bool {
        let bulk_size = self.bulk_keys.len().max(1);
        let buffer_ratio = self.buffer.len() as f64 / bulk_size as f64;

        if buffer_ratio >= self.rebuild_threshold {
            return true;
        }

        // Check for distribution shift via average prediction error.
        if self.recent_errors.len() >= self.error_window / 2 {
            let avg_error: f64 =
                self.recent_errors.iter().sum::<usize>() as f64 / self.recent_errors.len() as f64;
            // If the average observed error is more than 80% of the max error,
            // the distribution may have shifted.
            if avg_error > self.config.max_error as f64 * 0.8 {
                return true;
            }
        }

        false
    }

    /// Rebuild the learned index, merging the buffer into the bulk data.
    pub fn rebuild(&mut self) {
        // Merge buffer into bulk data (sorted merge).
        let mut merged_keys = Vec::with_capacity(self.bulk_keys.len() + self.buffer.len());
        let mut merged_values = Vec::with_capacity(self.bulk_keys.len() + self.buffer.len());

        let mut bulk_iter = self
            .bulk_keys
            .iter()
            .zip(self.bulk_values.iter())
            .peekable();
        let mut buf_iter = self.buffer.iter().peekable();

        loop {
            match (bulk_iter.peek(), buf_iter.peek()) {
                (Some(&(bk, _)), Some(&(uk, _))) => {
                    if bk == uk {
                        // Buffer value wins; skip bulk entry.
                        merged_keys.push(*uk);
                        merged_values.push(buf_iter.next().unwrap().1.clone());
                        bulk_iter.next();
                    } else if bk < uk {
                        merged_keys.push(*bk);
                        merged_values.push(bulk_iter.next().unwrap().1.clone());
                    } else {
                        merged_keys.push(*uk);
                        merged_values.push(buf_iter.next().unwrap().1.clone());
                    }
                }
                (Some(_), None) => {
                    let (k, v) = bulk_iter.next().unwrap();
                    merged_keys.push(*k);
                    merged_values.push(v.clone());
                }
                (None, Some(_)) => {
                    let (k, v) = buf_iter.next().unwrap();
                    merged_keys.push(*k);
                    merged_values.push(v.clone());
                }
                (None, None) => break,
            }
        }

        self.buffer.clear();
        self.bulk_load(merged_keys, merged_values);
    }

    /// Record a prediction error for distribution-shift detection.
    pub fn record_error(&mut self, error: usize) {
        if self.recent_errors.len() >= self.error_window {
            self.recent_errors.remove(0);
        }
        self.recent_errors.push(error);
    }

    /// Returns the number of entries in the B-Tree buffer.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the total number of entries (bulk + buffer).
    pub fn total_entries(&self) -> usize {
        self.bulk_keys.len() + self.buffer.len()
    }

    /// Returns `true` if the hybrid index contains no data.
    pub fn is_empty(&self) -> bool {
        self.bulk_keys.is_empty() && self.buffer.is_empty()
    }
}

impl Default for LearnedBTreeHybrid {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- RadixSplineConfig --------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = RadixSplineConfig::default();
        assert_eq!(config.max_error, 32);
        assert_eq!(config.num_radix_bits, 18);
        assert_eq!(config.spline_max_points, 1024);
    }

    // -- SplinePoint --------------------------------------------------------

    #[test]
    fn test_spline_point_equality() {
        let a = SplinePoint {
            key: 42,
            position: 7,
        };
        let b = SplinePoint {
            key: 42,
            position: 7,
        };
        let c = SplinePoint {
            key: 43,
            position: 7,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // -- RadixSpline build & lookup -----------------------------------------

    #[test]
    fn test_build_single_key() {
        let keys = vec![100u64];
        let positions = vec![0usize];
        let config = RadixSplineConfig::default();
        let rs = RadixSpline::build(&keys, &positions, config);

        assert_eq!(rs.min_key, 100);
        assert_eq!(rs.max_key, 100);
        assert!(rs.num_spline_points() >= 1);

        let (pos, err) = rs.lookup(100);
        assert_eq!(pos, 0);
        assert_eq!(err, 32);
    }

    #[test]
    fn test_build_uniform_distribution() {
        let n = 10_000usize;
        let keys: Vec<u64> = (0..n as u64).collect();
        let positions: Vec<usize> = (0..n).collect();

        let config = RadixSplineConfig {
            max_error: 16,
            num_radix_bits: 10, // smaller for test speed
            spline_max_points: 512,
        };
        let rs = RadixSpline::build(&keys, &positions, config);

        assert_eq!(rs.key_range(), (0, 9999));

        // Check that the spline is compact for uniform data.
        // Uniform distribution should need very few spline points.
        assert!(
            rs.num_spline_points() < 50,
            "Uniform data should produce few spline points, got {}",
            rs.num_spline_points()
        );
    }

    #[test]
    fn test_lookup_accuracy_uniform() {
        let n = 10_000usize;
        let keys: Vec<u64> = (0..n as u64).collect();
        let positions: Vec<usize> = (0..n).collect();

        let config = RadixSplineConfig {
            max_error: 16,
            num_radix_bits: 10,
            spline_max_points: 512,
        };
        let rs = RadixSpline::build(&keys, &positions, config);

        // Sample lookups: every 100th key.
        for i in (0..n).step_by(100) {
            let (est_pos, err) = rs.lookup(keys[i]);
            let actual_pos = positions[i];
            let diff = if est_pos > actual_pos {
                est_pos - actual_pos
            } else {
                actual_pos - est_pos
            };
            assert!(
                diff <= err,
                "Lookup for key {} at position {}: estimated {}, error {}, diff {}",
                keys[i],
                actual_pos,
                est_pos,
                err,
                diff
            );
        }
    }

    #[test]
    fn test_lookup_accuracy_quadratic() {
        // Non-uniform distribution: quadratic spacing.
        let n = 5_000usize;
        let keys: Vec<u64> = (0..n as u64).map(|i| i * i).collect();
        let positions: Vec<usize> = (0..n).collect();

        let config = RadixSplineConfig {
            max_error: 32,
            num_radix_bits: 10,
            spline_max_points: 1024,
        };
        let rs = RadixSpline::build(&keys, &positions, config);

        for i in (0..n).step_by(50) {
            let (est_pos, err) = rs.lookup(keys[i]);
            let actual_pos = positions[i];
            let diff = if est_pos > actual_pos {
                est_pos - actual_pos
            } else {
                actual_pos - est_pos
            };
            assert!(
                diff <= err,
                "Quadratic key {} at position {}: estimated {}, error {}, diff {}",
                keys[i],
                actual_pos,
                est_pos,
                err,
                diff
            );
        }
    }

    #[test]
    fn test_search_finds_existing_keys() {
        let n = 1_000usize;
        let keys: Vec<u64> = (0..n as u64).map(|i| i * 3 + 10).collect();
        let positions: Vec<usize> = (0..n).collect();

        let config = RadixSplineConfig {
            max_error: 16,
            num_radix_bits: 8,
            spline_max_points: 256,
        };
        let rs = RadixSpline::build(&keys, &positions, config);

        for i in (0..n).step_by(10) {
            let result = rs.search(keys[i], &keys);
            assert_eq!(
                result,
                Some(i),
                "search({}) should find index {}",
                keys[i],
                i
            );
        }
    }

    #[test]
    fn test_search_missing_keys() {
        let keys: Vec<u64> = (0..1000u64).map(|i| i * 2).collect(); // even numbers
        let positions: Vec<usize> = (0..1000).collect();

        let config = RadixSplineConfig {
            max_error: 16,
            num_radix_bits: 8,
            spline_max_points: 256,
        };
        let rs = RadixSpline::build(&keys, &positions, config);

        // Odd numbers don't exist.
        for i in (1..100).step_by(2) {
            let result = rs.search(i, &keys);
            assert_eq!(result, None, "search({}) should return None", i);
        }
    }

    #[test]
    fn test_search_empty_data() {
        let keys: Vec<u64> = vec![1, 2, 3];
        let positions: Vec<usize> = vec![0, 1, 2];
        let config = RadixSplineConfig::default();
        let rs = RadixSpline::build(&keys, &positions, config);

        let empty: Vec<u64> = Vec::new();
        assert_eq!(rs.search(1, &empty), None);
    }

    #[test]
    fn test_lookup_out_of_range_keys() {
        let keys: Vec<u64> = vec![100, 200, 300, 400, 500];
        let positions: Vec<usize> = vec![0, 1, 2, 3, 4];
        let config = RadixSplineConfig {
            max_error: 8,
            num_radix_bits: 4,
            spline_max_points: 64,
        };
        let rs = RadixSpline::build(&keys, &positions, config);

        // Key below minimum should still return a valid estimate.
        let (pos, _) = rs.lookup(0);
        assert!(pos <= 4, "Position for key 0 should be in valid range");

        // Key above maximum should still return a valid estimate.
        let (pos, _) = rs.lookup(1000);
        assert!(pos <= 4, "Position for key 1000 should be in valid range");
    }

    #[test]
    fn test_duplicate_keys() {
        let keys: Vec<u64> = vec![1, 1, 2, 2, 2, 3, 3, 4, 5, 5];
        let positions: Vec<usize> = (0..10).collect();
        let config = RadixSplineConfig {
            max_error: 8,
            num_radix_bits: 4,
            spline_max_points: 64,
        };
        let rs = RadixSpline::build(&keys, &positions, config);

        // Should find at least one occurrence.
        let result = rs.search(2, &keys);
        assert!(result.is_some());
        assert_eq!(keys[result.unwrap()], 2);
    }

    #[test]
    fn test_memory_usage() {
        let keys: Vec<u64> = (0..1000u64).collect();
        let positions: Vec<usize> = (0..1000).collect();
        let config = RadixSplineConfig {
            max_error: 32,
            num_radix_bits: 8,
            spline_max_points: 64,
        };
        let rs = RadixSpline::build(&keys, &positions, config);

        let usage = rs.memory_usage();
        // Radix table: 256 * 8 bytes = 2048 bytes minimum.
        assert!(usage > 0);
        assert!(usage < 1_000_000, "Memory usage should be bounded");
    }

    #[test]
    fn test_small_radix_bits() {
        let keys: Vec<u64> = (0..100u64).collect();
        let positions: Vec<usize> = (0..100).collect();
        let config = RadixSplineConfig {
            max_error: 8,
            num_radix_bits: 2, // only 4 buckets
            spline_max_points: 64,
        };
        let rs = RadixSpline::build(&keys, &positions, config);

        for i in 0..100 {
            let (est, err) = rs.lookup(i);
            let diff = if est > i as usize {
                est - i as usize
            } else {
                i as usize - est
            };
            assert!(
                diff <= err,
                "Key {}: est={}, err={}, diff={}",
                i,
                est,
                err,
                diff
            );
        }
    }

    // -- LearnedBTreeHybrid -------------------------------------------------

    #[test]
    fn test_hybrid_new_is_empty() {
        let hybrid = LearnedBTreeHybrid::new();
        assert!(hybrid.is_empty());
        assert_eq!(hybrid.total_entries(), 0);
        assert_eq!(hybrid.buffer_size(), 0);
    }

    #[test]
    fn test_hybrid_insert_and_get() {
        let mut hybrid = LearnedBTreeHybrid::new();

        hybrid.insert(10, b"hello".to_vec());
        hybrid.insert(20, b"world".to_vec());

        assert_eq!(hybrid.get(10), Some(b"hello".as_slice()));
        assert_eq!(hybrid.get(20), Some(b"world".as_slice()));
        assert_eq!(hybrid.get(15), None);
        assert_eq!(hybrid.buffer_size(), 2);
    }

    #[test]
    fn test_hybrid_bulk_load_and_get() {
        let mut hybrid = LearnedBTreeHybrid::new();

        let keys: Vec<u64> = (0..100).collect();
        let values: Vec<Vec<u8>> = (0..100)
            .map(|i| format!("val_{}", i).into_bytes())
            .collect();

        hybrid.bulk_load(keys, values);

        assert_eq!(hybrid.total_entries(), 100);
        assert_eq!(hybrid.get(0), Some(b"val_0".as_slice()));
        assert_eq!(hybrid.get(50), Some(b"val_50".as_slice()));
        assert_eq!(hybrid.get(99), Some(b"val_99".as_slice()));
        assert_eq!(hybrid.get(100), None);
    }

    #[test]
    fn test_hybrid_buffer_overrides_bulk() {
        let mut hybrid = LearnedBTreeHybrid::new();

        let keys: Vec<u64> = vec![10, 20, 30];
        let values: Vec<Vec<u8>> = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        hybrid.bulk_load(keys, values);

        // Override key 20 via buffer.
        hybrid.insert(20, b"new_b".to_vec());

        assert_eq!(hybrid.get(20), Some(b"new_b".as_slice()));
        assert_eq!(hybrid.get(10), Some(b"a".as_slice()));
    }

    #[test]
    fn test_hybrid_rebuild() {
        let mut hybrid = LearnedBTreeHybrid::new();

        let keys: Vec<u64> = vec![10, 30, 50];
        let values: Vec<Vec<u8>> = vec![b"a".to_vec(), b"c".to_vec(), b"e".to_vec()];
        hybrid.bulk_load(keys, values);

        // Insert into buffer.
        hybrid.insert(20, b"b".to_vec());
        hybrid.insert(40, b"d".to_vec());

        assert_eq!(hybrid.buffer_size(), 2);

        // Rebuild merges buffer into bulk.
        hybrid.rebuild();

        assert_eq!(hybrid.buffer_size(), 0);
        assert_eq!(hybrid.total_entries(), 5);
        assert_eq!(hybrid.get(10), Some(b"a".as_slice()));
        assert_eq!(hybrid.get(20), Some(b"b".as_slice()));
        assert_eq!(hybrid.get(30), Some(b"c".as_slice()));
        assert_eq!(hybrid.get(40), Some(b"d".as_slice()));
        assert_eq!(hybrid.get(50), Some(b"e".as_slice()));
    }

    #[test]
    fn test_hybrid_rebuild_with_override() {
        let mut hybrid = LearnedBTreeHybrid::new();

        let keys: Vec<u64> = vec![10, 20, 30];
        let values: Vec<Vec<u8>> = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        hybrid.bulk_load(keys, values);

        // Override key 20 in buffer, then rebuild.
        hybrid.insert(20, b"B_NEW".to_vec());
        hybrid.rebuild();

        assert_eq!(hybrid.get(20), Some(b"B_NEW".as_slice()));
        assert_eq!(hybrid.total_entries(), 3);
    }

    #[test]
    fn test_hybrid_needs_rebuild_by_buffer_size() {
        let mut hybrid = LearnedBTreeHybrid::new();

        let keys: Vec<u64> = (0..100).collect();
        let values: Vec<Vec<u8>> = (0..100).map(|i| vec![i as u8]).collect();
        hybrid.bulk_load(keys, values);

        assert!(!hybrid.needs_rebuild());

        // Insert 20% of bulk size (threshold is 0.2).
        for i in 100..120 {
            hybrid.insert(i, vec![i as u8]);
        }

        assert!(hybrid.needs_rebuild());
    }

    #[test]
    fn test_hybrid_needs_rebuild_by_error_shift() {
        let mut hybrid = LearnedBTreeHybrid::new();

        let keys: Vec<u64> = (0..100).collect();
        let values: Vec<Vec<u8>> = (0..100).map(|i| vec![i as u8]).collect();
        hybrid.bulk_load(keys, values);

        // Simulate high prediction errors (distribution shift).
        for _ in 0..600 {
            hybrid.record_error(30); // 30 out of max_error=32 -> 93.75% > 80%
        }

        assert!(hybrid.needs_rebuild());
    }

    #[test]
    fn test_hybrid_no_rebuild_with_low_errors() {
        let mut hybrid = LearnedBTreeHybrid::new();

        let keys: Vec<u64> = (0..100).collect();
        let values: Vec<Vec<u8>> = (0..100).map(|i| vec![i as u8]).collect();
        hybrid.bulk_load(keys, values);

        // Low errors: well within bounds.
        for _ in 0..600 {
            hybrid.record_error(5);
        }

        assert!(!hybrid.needs_rebuild());
    }

    #[test]
    fn test_hybrid_default_impl() {
        let hybrid = LearnedBTreeHybrid::default();
        assert!(hybrid.is_empty());
    }

    #[test]
    fn test_hybrid_empty_bulk_load() {
        let mut hybrid = LearnedBTreeHybrid::new();
        hybrid.bulk_load(Vec::new(), Vec::new());
        assert!(hybrid.is_empty());
        assert_eq!(hybrid.get(42), None);
    }

    // -- Stress / larger dataset tests --------------------------------------

    #[test]
    fn test_large_dataset_lookup_correctness() {
        let n = 50_000usize;
        let keys: Vec<u64> = (0..n as u64).map(|i| i * 7 + 1000).collect();
        let positions: Vec<usize> = (0..n).collect();

        let config = RadixSplineConfig {
            max_error: 64,
            num_radix_bits: 12,
            spline_max_points: 1024,
        };
        let rs = RadixSpline::build(&keys, &positions, config);

        // Verify every 500th key.
        let mut violations = 0;
        for i in (0..n).step_by(500) {
            let (est, err) = rs.lookup(keys[i]);
            let diff = if est > i { est - i } else { i - est };
            if diff > err {
                violations += 1;
            }
        }
        assert_eq!(violations, 0, "No error bound violations allowed");
    }

    #[test]
    fn test_large_dataset_search_correctness() {
        let n = 10_000usize;
        let keys: Vec<u64> = (0..n as u64).map(|i| i * 5).collect();
        let positions: Vec<usize> = (0..n).collect();

        let config = RadixSplineConfig {
            max_error: 32,
            num_radix_bits: 10,
            spline_max_points: 512,
        };
        let rs = RadixSpline::build(&keys, &positions, config);

        // Every existing key must be found.
        for i in (0..n).step_by(100) {
            assert_eq!(
                rs.search(keys[i], &keys),
                Some(i),
                "Must find key {} at index {}",
                keys[i],
                i
            );
        }

        // Non-existing keys (values not divisible by 5 are not in the set).
        for i in (1..100).filter(|x| x % 5 != 0) {
            assert_eq!(rs.search(i, &keys), None, "Key {} should not be found", i);
        }
    }

    #[test]
    fn test_spline_compactness_uniform() {
        // Perfectly uniform data should be representable with very few points.
        let n = 100_000usize;
        let keys: Vec<u64> = (0..n as u64).collect();
        let positions: Vec<usize> = (0..n).collect();

        let config = RadixSplineConfig {
            max_error: 1,
            num_radix_bits: 10,
            spline_max_points: 1024,
        };
        let rs = RadixSpline::build(&keys, &positions, config);

        // For perfectly linear data with error=1, we should need only 2 points
        // (start and end), or very few.
        assert!(
            rs.num_spline_points() <= 10,
            "Uniform data with error=1 should need very few points, got {}",
            rs.num_spline_points()
        );
    }

    #[test]
    fn test_hybrid_full_workflow() {
        let mut hybrid = LearnedBTreeHybrid::with_config(RadixSplineConfig {
            max_error: 16,
            num_radix_bits: 8,
            spline_max_points: 64,
        });

        // Phase 1: Bulk load.
        let keys: Vec<u64> = (0..500u64).map(|i| i * 2).collect();
        let values: Vec<Vec<u8>> = keys
            .iter()
            .map(|k| format!("v{}", k).into_bytes())
            .collect();
        hybrid.bulk_load(keys, values);

        assert_eq!(hybrid.total_entries(), 500);

        // Phase 2: Buffer inserts.
        for i in 0..50 {
            let key = i * 2 + 1; // odd numbers
            hybrid.insert(key, format!("buf{}", key).into_bytes());
        }

        assert_eq!(hybrid.buffer_size(), 50);

        // Phase 3: Read from both layers.
        assert_eq!(hybrid.get(0), Some(b"v0".as_slice()));
        assert_eq!(hybrid.get(1), Some(b"buf1".as_slice()));
        assert_eq!(hybrid.get(998), Some(b"v998".as_slice()));

        // Phase 4: Rebuild.
        hybrid.rebuild();
        assert_eq!(hybrid.buffer_size(), 0);
        assert_eq!(hybrid.total_entries(), 550);

        // Data still accessible after rebuild.
        assert_eq!(hybrid.get(0), Some(b"v0".as_slice()));
        assert_eq!(hybrid.get(1), Some(b"buf1".as_slice()));
    }
}
