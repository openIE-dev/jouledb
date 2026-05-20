//! Columnar Projections for OLAP-style Analytics
//!
//! This module provides write-time columnar encoding to enable fast analytical queries.
//! Instead of extracting values from documents at query time, values are pre-projected
//! into contiguous arrays during ingestion.
//!
//! ## Architecture
//!
//! ```text
//! Document Ingest:
//!   {"l_extendedprice": 1234.56, "l_quantity": 10, "l_shipdate": 19950315}
//!        |                           |                    |
//!        v                           v                    v
//!   columns["l_extendedprice"]  columns["l_quantity"]  columns["l_shipdate"]
//!   [1234.56, ...]              [10.0, ...]            [19950315.0, ...]
//! ```
//!
//! ## Performance
//!
//! - **Scan**: O(n) with SIMD vectorization potential
//! - **Aggregate**: O(n) but cache-friendly contiguous access
//! - **Memory**: ~8 bytes per numeric value per field

use crate::{RecordId, Value};
use std::collections::{HashMap, HashSet};

/// Columnar storage for a single field
#[derive(Debug, Clone)]
pub struct Column {
    /// Contiguous array of values (f64 for all numerics)
    pub values: Vec<f64>,
    /// Parallel array of record IDs for mapping back to documents
    pub record_ids: Vec<RecordId>,
    /// Min value seen (for statistics/pruning)
    pub min: f64,
    /// Max value seen (for statistics/pruning)
    pub max: f64,
    /// Running sum (for fast AVG computation)
    pub sum: f64,
    /// Tombstones for deleted records (lazy deletion)
    tombstones: HashSet<RecordId>,
    /// Pending delta updates (record_id -> new_value)
    deltas: Vec<(RecordId, f64)>,
    /// Whether statistics need recomputation
    stats_dirty: bool,
}

impl Column {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            record_ids: Vec::new(),
            min: f64::MAX,
            max: f64::MIN,
            sum: 0.0,
            tombstones: HashSet::new(),
            deltas: Vec::new(),
            stats_dirty: false,
        }
    }

    /// Append a value to this column
    #[inline]
    pub fn push(&mut self, record_id: RecordId, value: f64) {
        self.values.push(value);
        self.record_ids.push(record_id);
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value;
    }

    /// Remove a record from this column (e.g., when value becomes NULL).
    pub fn remove_record(&mut self, record_id: RecordId) {
        if let Some(pos) = self.record_ids.iter().position(|&id| id == record_id) {
            let old_value = self.values[pos];
            self.values.remove(pos);
            self.record_ids.remove(pos);
            self.sum -= old_value;
            if old_value == self.min || old_value == self.max {
                self.stats_dirty = true;
            }
        }
    }

    /// Update a value for a specific record in this column.
    ///
    /// If the record exists, its value is updated in-place.
    /// If the record doesn't exist, the value is appended.
    pub fn update(&mut self, record_id: RecordId, value: f64) {
        // Try to find the record in the column
        if let Some(pos) = self.record_ids.iter().position(|&id| id == record_id) {
            // Record exists - update in place
            let old_value = self.values[pos];
            self.values[pos] = value;

            // Adjust sum
            self.sum -= old_value;
            self.sum += value;

            // Mark stats as dirty if min/max might have changed
            if old_value == self.min
                || old_value == self.max
                || value < self.min
                || value > self.max
            {
                self.stats_dirty = true;
                // Eagerly update min/max if possible
                self.min = self.min.min(value);
                self.max = self.max.max(value);
            }
        } else {
            // Record doesn't exist - append
            self.push(record_id, value);
        }
    }

    /// Recompute min/max if stats are dirty
    pub fn refresh_stats(&mut self) {
        if !self.stats_dirty {
            return;
        }

        self.min = f64::MAX;
        self.max = f64::MIN;
        for (i, &value) in self.values.iter().enumerate() {
            if !self.tombstones.contains(&(i as RecordId)) {
                self.min = self.min.min(value);
                self.max = self.max.max(value);
            }
        }
        self.stats_dirty = false;
    }

    /// Number of values in this column
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get pre-computed sum (O(1))
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Get count (O(1))
    #[inline]
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// Get average (O(1))
    #[inline]
    pub fn avg(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }

    /// Get min (O(1))
    #[inline]
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Get max (O(1))
    #[inline]
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Sum values where predicate matches (vectorizable)
    pub fn sum_where<F: Fn(f64) -> bool>(&self, predicate: F) -> f64 {
        self.values.iter().filter(|&&v| predicate(v)).sum()
    }

    /// Count values where predicate matches
    pub fn count_where<F: Fn(f64) -> bool>(&self, predicate: F) -> usize {
        self.values.iter().filter(|&&v| predicate(v)).count()
    }

    /// Sum values in range [min, max)
    pub fn sum_range(&self, min: f64, max: f64) -> f64 {
        // Early exit if range doesn't overlap
        if max <= self.min || min > self.max {
            return 0.0;
        }

        self.values.iter().filter(|&&v| v >= min && v < max).sum()
    }

    /// Count values in range [min, max)
    pub fn count_range(&self, min: f64, max: f64) -> usize {
        // Early exit if range doesn't overlap
        if max <= self.min || min > self.max {
            return 0;
        }

        self.values.iter().filter(|&&v| v >= min && v < max).count()
    }

    /// Get record IDs where value is in range [min, max)
    pub fn filter_range(&self, min: f64, max: f64) -> Vec<RecordId> {
        // Early exit if range doesn't overlap
        if max <= self.min || min > self.max {
            return Vec::new();
        }

        self.values
            .iter()
            .zip(self.record_ids.iter())
            .filter(|(v, id)| **v >= min && **v < max && !self.tombstones.contains(id))
            .map(|(_, id)| *id)
            .collect()
    }

    // =========================================================================
    // Tombstone / Incremental Maintenance
    // =========================================================================

    /// Mark a record as deleted (lazy deletion via tombstone)
    pub fn mark_deleted(&mut self, record_id: RecordId) {
        self.tombstones.insert(record_id);
        self.stats_dirty = true;
    }

    /// Check if a record is deleted
    pub fn is_deleted(&self, record_id: RecordId) -> bool {
        self.tombstones.contains(&record_id)
    }

    /// Get count of tombstoned records
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    /// Get value for a specific record ID (O(n) scan, use sparingly)
    pub fn get_value(&self, record_id: RecordId) -> Option<f64> {
        if self.tombstones.contains(&record_id) {
            return None;
        }
        self.record_ids
            .iter()
            .position(|&id| id == record_id)
            .map(|idx| self.values[idx])
    }

    /// Compute live sum (accounting for tombstones)
    pub fn sum_live(&self) -> f64 {
        if self.tombstones.is_empty() {
            return self.sum;
        }
        self.values
            .iter()
            .zip(self.record_ids.iter())
            .filter(|&(_, id)| !self.tombstones.contains(&id))
            .map(|(&v, _)| v)
            .sum()
    }

    /// Compute live count (accounting for tombstones)
    pub fn count_live(&self) -> usize {
        self.values.len() - self.tombstones.len()
    }

    /// Compute live average (accounting for tombstones)
    pub fn avg_live(&self) -> f64 {
        let count = self.count_live();
        if count == 0 {
            0.0
        } else {
            self.sum_live() / count as f64
        }
    }

    /// Compact the column by removing tombstoned entries
    /// Call this periodically to reclaim space and restore O(1) statistics
    pub fn compact(&mut self) {
        if self.tombstones.is_empty() {
            return;
        }

        // Filter out tombstoned entries
        let mut new_values = Vec::with_capacity(self.values.len() - self.tombstones.len());
        let mut new_record_ids = Vec::with_capacity(self.record_ids.len() - self.tombstones.len());
        let mut new_sum = 0.0;
        let mut new_min = f64::MAX;
        let mut new_max = f64::MIN;

        for (value, record_id) in self.values.iter().zip(self.record_ids.iter()) {
            if !self.tombstones.contains(record_id) {
                new_values.push(*value);
                new_record_ids.push(*record_id);
                new_sum += value;
                new_min = new_min.min(*value);
                new_max = new_max.max(*value);
            }
        }

        self.values = new_values;
        self.record_ids = new_record_ids;
        self.sum = new_sum;
        self.min = new_min;
        self.max = new_max;
        self.tombstones.clear();
        self.stats_dirty = false;
    }

    /// Check if compaction is recommended (>10% tombstones)
    pub fn needs_compaction(&self) -> bool {
        if self.values.is_empty() {
            return false;
        }
        self.tombstones.len() * 10 > self.values.len()
    }

    // =========================================================================
    // Join Support
    // =========================================================================

    /// Build a hash index for join operations
    /// Returns: value -> list of record IDs with that value
    pub fn build_hash_index(&self) -> HashMap<i64, Vec<RecordId>> {
        let mut index: HashMap<i64, Vec<RecordId>> = HashMap::new();
        for (&value, &record_id) in self.values.iter().zip(self.record_ids.iter()) {
            if !self.tombstones.contains(&record_id) {
                let key = value as i64; // Convert to integer key for exact matching
                index.entry(key).or_default().push(record_id);
            }
        }
        index
    }

    /// Scan column values with record IDs (for join probe phase)
    pub fn scan(&self) -> impl Iterator<Item = (RecordId, f64)> + '_ {
        self.record_ids
            .iter()
            .zip(self.values.iter())
            .filter(|&(id, _)| !self.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
    }

    // =========================================================================
    // Histogram Statistics
    // =========================================================================

    /// Build an equi-height histogram for more accurate selectivity estimation
    ///
    /// # Arguments
    /// * `num_buckets` - Number of histogram buckets to create
    ///
    /// # Returns
    /// A Histogram with the specified number of buckets
    pub fn build_histogram(&self, num_buckets: usize) -> Histogram {
        let num_buckets = num_buckets.max(1);

        // Collect non-tombstoned values
        let mut values: Vec<f64> = self
            .values
            .iter()
            .zip(self.record_ids.iter())
            .filter(|&(_, id)| !self.tombstones.contains(&id))
            .map(|(&v, _)| v)
            .collect();

        if values.is_empty() {
            return Histogram {
                buckets: Vec::new(),
                ndv: 0,
                total_count: 0,
            };
        }

        // Sort for percentile calculation
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Count distinct values
        let mut ndv = 1;
        for i in 1..values.len() {
            if (values[i] - values[i - 1]).abs() > f64::EPSILON {
                ndv += 1;
            }
        }

        let total_count = values.len();
        let bucket_size = (values.len() + num_buckets - 1) / num_buckets;

        let mut buckets = Vec::with_capacity(num_buckets);
        let mut i = 0;

        while i < values.len() {
            let bucket_end = (i + bucket_size).min(values.len());
            let bucket_values = &values[i..bucket_end];

            let lower = bucket_values[0];
            let upper = bucket_values[bucket_values.len() - 1];
            let count = bucket_values.len();

            // Count distinct in this bucket
            let mut bucket_distinct = 1;
            for j in 1..bucket_values.len() {
                if (bucket_values[j] - bucket_values[j - 1]).abs() > f64::EPSILON {
                    bucket_distinct += 1;
                }
            }

            buckets.push(HistogramBucket {
                lower,
                upper,
                count,
                distinct_count: bucket_distinct,
            });

            i = bucket_end;
        }

        Histogram {
            buckets,
            ndv,
            total_count,
        }
    }

    /// Estimate selectivity using histogram (more accurate than uniform assumption)
    pub fn estimate_selectivity_with_histogram(
        &self,
        min: f64,
        max: f64,
        histogram: &Histogram,
    ) -> f64 {
        if histogram.total_count == 0 {
            return 0.0;
        }

        let mut selected_count = 0.0;

        for bucket in &histogram.buckets {
            // Check overlap between filter range and bucket
            if max <= bucket.lower || min > bucket.upper {
                // No overlap
                continue;
            }

            if min <= bucket.lower && max > bucket.upper {
                // Bucket fully contained in range
                selected_count += bucket.count as f64;
            } else {
                // Partial overlap - estimate using uniform distribution within bucket
                let bucket_range = bucket.upper - bucket.lower;
                if bucket_range <= f64::EPSILON {
                    // Single-value bucket
                    if bucket.lower >= min && bucket.lower < max {
                        selected_count += bucket.count as f64;
                    }
                } else {
                    let overlap_min = min.max(bucket.lower);
                    let overlap_max = max.min(bucket.upper + f64::EPSILON);
                    let overlap_ratio = (overlap_max - overlap_min) / bucket_range;
                    selected_count += bucket.count as f64 * overlap_ratio.clamp(0.0, 1.0);
                }
            }
        }

        (selected_count / histogram.total_count as f64).clamp(0.0, 1.0)
    }
}

/// Histogram bucket for statistics
#[derive(Debug, Clone)]
pub struct HistogramBucket {
    /// Lower bound of the bucket (inclusive)
    pub lower: f64,
    /// Upper bound of the bucket (inclusive)
    pub upper: f64,
    /// Number of values in this bucket
    pub count: usize,
    /// Number of distinct values in this bucket
    pub distinct_count: usize,
}

/// Histogram for selectivity estimation
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Equi-height buckets
    pub buckets: Vec<HistogramBucket>,
    /// Total number of distinct values
    pub ndv: usize,
    /// Total count of values
    pub total_count: usize,
}

impl Histogram {
    /// Get the estimated selectivity for an equality predicate
    pub fn selectivity_equals(&self, value: f64) -> f64 {
        if self.total_count == 0 || self.ndv == 0 {
            return 0.0;
        }

        // Find the bucket containing the value
        for bucket in &self.buckets {
            if value >= bucket.lower && value <= bucket.upper {
                // Assume uniform distribution within bucket
                if bucket.distinct_count > 0 {
                    return (bucket.count as f64 / bucket.distinct_count as f64)
                        / self.total_count as f64;
                }
            }
        }

        // Value not in any bucket - assume minimum selectivity
        1.0 / self.total_count as f64
    }

    /// Get the estimated selectivity for a range predicate
    pub fn selectivity_range(&self, min: f64, max: f64) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        let mut selected = 0.0;

        for bucket in &self.buckets {
            if max <= bucket.lower || min > bucket.upper {
                continue;
            }

            if min <= bucket.lower && max > bucket.upper {
                selected += bucket.count as f64;
            } else {
                let bucket_range = bucket.upper - bucket.lower;
                if bucket_range <= f64::EPSILON {
                    if bucket.lower >= min && bucket.lower < max {
                        selected += bucket.count as f64;
                    }
                } else {
                    let overlap = (max.min(bucket.upper + f64::EPSILON) - min.max(bucket.lower))
                        / bucket_range;
                    selected += bucket.count as f64 * overlap.clamp(0.0, 1.0);
                }
            }
        }

        (selected / self.total_count as f64).clamp(0.0, 1.0)
    }
}

impl Default for Column {
    fn default() -> Self {
        Self::new()
    }
}

/// Columnar store - maintains columnar projections for all numeric fields
#[derive(Debug, Clone, Default)]
pub struct ColumnarStore {
    /// Columns indexed by field name
    columns: HashMap<String, Column>,
}

impl ColumnarStore {
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
        }
    }

    /// Record a numeric value from a document
    pub fn record_value(&mut self, field: &str, record_id: RecordId, value: &Value) {
        let numeric_value = match value {
            Value::Int(i) => *i as f64,
            Value::Float(f) => *f,
            _ => return, // Skip non-numeric values
        };

        self.columns
            .entry(field.to_string())
            .or_insert_with(Column::new)
            .push(record_id, numeric_value);
    }

    /// Get a column by name
    pub fn get_column(&self, field: &str) -> Option<&Column> {
        self.columns.get(field)
    }

    /// Get mutable column by name
    pub fn get_column_mut(&mut self, field: &str) -> Option<&mut Column> {
        self.columns.get_mut(field)
    }

    /// List all column names
    pub fn column_names(&self) -> impl Iterator<Item = &String> {
        self.columns.keys()
    }

    /// Get statistics for all columns
    pub fn stats(&self) -> ColumnarStats {
        let mut total_values = 0;
        let mut columns = Vec::new();

        for (name, col) in &self.columns {
            total_values += col.len();
            columns.push(ColumnStats {
                name: name.clone(),
                count: col.len(),
                min: col.min,
                max: col.max,
                sum: col.sum,
            });
        }

        ColumnarStats {
            num_columns: self.columns.len(),
            total_values,
            columns,
        }
    }

    // =========================================================================
    // Incremental Updates
    // =========================================================================

    /// Update values for a specific record in columnar storage.
    ///
    /// This updates values in-place without rebuilding the entire column.
    /// For each field in updates:
    /// - If it's a new numeric field, it's added to the column
    /// - If it already exists, the value is updated
    pub fn update_values(&mut self, record_id: RecordId, updates: &HashMap<String, Value>) {
        for (field, value) in updates {
            // NULL means the value was set to NULL — remove from columnar
            if matches!(value, Value::Null) {
                if let Some(col) = self.columns.get_mut(field) {
                    col.remove_record(record_id);
                }
                continue;
            }

            let numeric_value = match value {
                Value::Int(i) => *i as f64,
                Value::Float(f) => *f,
                _ => continue, // Skip non-numeric values
            };

            // Get or create the column
            let col = self
                .columns
                .entry(field.clone())
                .or_insert_with(Column::new);

            // Update the value for this record
            col.update(record_id, numeric_value);
        }
    }

    // =========================================================================
    // Aggregate Query Methods
    // =========================================================================

    /// SUM of a column
    pub fn sum(&self, field: &str) -> Option<f64> {
        self.columns.get(field).map(|c| c.sum())
    }

    /// COUNT of a column
    pub fn count(&self, field: &str) -> Option<usize> {
        self.columns.get(field).map(|c| c.count())
    }

    /// AVG of a column
    pub fn avg(&self, field: &str) -> Option<f64> {
        self.columns.get(field).map(|c| c.avg())
    }

    /// MIN of a column
    pub fn min(&self, field: &str) -> Option<f64> {
        self.columns.get(field).map(|c| c.min())
    }

    /// MAX of a column
    pub fn max(&self, field: &str) -> Option<f64> {
        self.columns.get(field).map(|c| c.max())
    }

    /// SUM where filter column is in range
    /// e.g., SUM(l_extendedprice) WHERE l_shipdate >= 19940101 AND l_shipdate < 19950101
    pub fn sum_where_range(
        &self,
        sum_field: &str,
        filter_field: &str,
        min: f64,
        max: f64,
    ) -> Option<f64> {
        let sum_col = self.columns.get(sum_field)?;
        let filter_col = self.columns.get(filter_field)?;

        // Columns must have same length (aligned by record order)
        if sum_col.len() != filter_col.len() {
            return None;
        }

        let result: f64 = sum_col
            .values
            .iter()
            .zip(filter_col.values.iter())
            .filter(|(_, filter_val)| **filter_val >= min && **filter_val < max)
            .map(|(sum_val, _)| *sum_val)
            .sum();

        Some(result)
    }

    /// COUNT where filter column is in range
    pub fn count_where_range(&self, filter_field: &str, min: f64, max: f64) -> Option<usize> {
        self.columns
            .get(filter_field)
            .map(|c| c.count_range(min, max))
    }

    // =========================================================================
    // Join Operations
    // =========================================================================

    /// Hash join between two columns (equi-join on integer keys)
    ///
    /// This implements a classic hash join:
    /// 1. Build phase: Create hash table from the build column
    /// 2. Probe phase: Scan probe column and look up matches
    ///
    /// # Arguments
    /// * `build_field` - The column to build hash table from (typically smaller)
    /// * `probe_field` - The column to probe against (typically larger)
    ///
    /// # Returns
    /// JoinResult containing matched (build_record_id, probe_record_id) pairs
    pub fn hash_join(&self, build_field: &str, probe_field: &str) -> Option<JoinResult> {
        let build_col = self.columns.get(build_field)?;
        let probe_col = self.columns.get(probe_field)?;

        // Build phase: create hash index from build column
        let hash_index = build_col.build_hash_index();

        // Probe phase: scan probe column and find matches
        let mut matches = Vec::new();

        for (probe_record_id, probe_value) in probe_col.scan() {
            let key = probe_value as i64;
            if let Some(build_record_ids) = hash_index.get(&key) {
                for &build_record_id in build_record_ids {
                    matches.push((build_record_id, probe_record_id));
                }
            }
        }

        Some(JoinResult {
            matches,
            build_field: build_field.to_string(),
            probe_field: probe_field.to_string(),
        })
    }

    /// Hash join with aggregation (SUM) on a value column
    ///
    /// Performs join and computes SUM of values from the probe side.
    /// Useful for TPC-H style queries like:
    /// SELECT SUM(l_extendedprice) FROM lineitem, orders WHERE l_orderkey = o_orderkey
    pub fn hash_join_sum(
        &self,
        build_field: &str,
        probe_field: &str,
        sum_field: &str,
    ) -> Option<f64> {
        let build_col = self.columns.get(build_field)?;
        let probe_col = self.columns.get(probe_field)?;
        let sum_col = self.columns.get(sum_field)?;

        // Build phase
        let hash_index = build_col.build_hash_index();

        // Create a map from probe record_id to sum value for fast lookup
        let sum_values: HashMap<RecordId, f64> = sum_col
            .record_ids
            .iter()
            .zip(sum_col.values.iter())
            .filter(|&(id, _)| !sum_col.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
            .collect();

        // Probe phase with aggregation
        let mut total = 0.0;

        for (probe_record_id, probe_value) in probe_col.scan() {
            let key = probe_value as i64;
            if hash_index.contains_key(&key) {
                // Match found - add the sum value for this probe record
                if let Some(&value) = sum_values.get(&probe_record_id) {
                    total += value;
                }
            }
        }

        Some(total)
    }

    /// Left outer join - returns all build records with optional probe matches
    ///
    /// For each build record, returns (build_id, Some(probe_id)) if match exists,
    /// or (build_id, None) if no match found.
    pub fn left_outer_join(
        &self,
        build_field: &str,
        probe_field: &str,
    ) -> Option<Vec<(RecordId, Option<RecordId>)>> {
        let build_col = self.columns.get(build_field)?;
        let probe_col = self.columns.get(probe_field)?;

        // Build hash index from probe side
        let probe_index = probe_col.build_hash_index();

        let mut results = Vec::new();

        for (build_record_id, build_value) in build_col.scan() {
            let key = build_value as i64;
            if let Some(probe_record_ids) = probe_index.get(&key) {
                // Matches found - emit one row per match
                for &probe_id in probe_record_ids {
                    results.push((build_record_id, Some(probe_id)));
                }
            } else {
                // No match - emit NULL on probe side
                results.push((build_record_id, None));
            }
        }

        Some(results)
    }

    /// Semi-join (EXISTS) - returns build records that have at least one probe match
    ///
    /// Equivalent to: SELECT * FROM build WHERE EXISTS (SELECT 1 FROM probe WHERE build.key = probe.key)
    pub fn semi_join(&self, build_field: &str, probe_field: &str) -> Option<Vec<RecordId>> {
        let build_col = self.columns.get(build_field)?;
        let probe_col = self.columns.get(probe_field)?;

        // Build hash set from probe side (we only need existence, not full index)
        let probe_keys: std::collections::HashSet<i64> =
            probe_col.scan().map(|(_, v)| v as i64).collect();

        let results: Vec<RecordId> = build_col
            .scan()
            .filter(|(_, v)| probe_keys.contains(&(*v as i64)))
            .map(|(id, _)| id)
            .collect();

        Some(results)
    }

    /// Anti-join (NOT EXISTS) - returns build records with no probe match
    ///
    /// Equivalent to: SELECT * FROM build WHERE NOT EXISTS (SELECT 1 FROM probe WHERE build.key = probe.key)
    pub fn anti_join(&self, build_field: &str, probe_field: &str) -> Option<Vec<RecordId>> {
        let build_col = self.columns.get(build_field)?;
        let probe_col = self.columns.get(probe_field)?;

        // Build hash set from probe side
        let probe_keys: std::collections::HashSet<i64> =
            probe_col.scan().map(|(_, v)| v as i64).collect();

        let results: Vec<RecordId> = build_col
            .scan()
            .filter(|(_, v)| !probe_keys.contains(&(*v as i64)))
            .map(|(id, _)| id)
            .collect();

        Some(results)
    }

    /// N-way hash join - joins multiple tables in sequence
    ///
    /// Takes pairs of (left_field, right_field) and joins them left to right.
    /// Returns record IDs that satisfy all join conditions.
    pub fn multi_way_join(&self, join_pairs: &[(String, String)]) -> Option<Vec<RecordId>> {
        if join_pairs.is_empty() {
            return None;
        }

        // Start with the first join
        let (first_build, first_probe) = &join_pairs[0];
        let mut current_ids: std::collections::HashSet<RecordId> = self
            .semi_join(first_build, first_probe)?
            .into_iter()
            .collect();

        // Apply subsequent joins as filters
        for (build_field, probe_field) in join_pairs.iter().skip(1) {
            let probe_col = self.columns.get(probe_field)?;
            let probe_keys: std::collections::HashSet<i64> =
                probe_col.scan().map(|(_, v)| v as i64).collect();

            if let Some(build_col) = self.columns.get(build_field) {
                current_ids = build_col
                    .scan()
                    .filter(|(id, v)| current_ids.contains(id) && probe_keys.contains(&(*v as i64)))
                    .map(|(id, _)| id)
                    .collect();
            }
        }

        Some(current_ids.into_iter().collect())
    }

    // =========================================================================
    // GROUP BY Aggregations
    // =========================================================================

    /// GROUP BY with SUM aggregation
    ///
    /// Returns HashMap<group_key, sum_value>
    pub fn group_by_sum(&self, group_field: &str, sum_field: &str) -> Option<HashMap<i64, f64>> {
        let group_col = self.columns.get(group_field)?;
        let sum_col = self.columns.get(sum_field)?;

        // Build map from record_id to sum value
        let sum_values: HashMap<RecordId, f64> = sum_col
            .record_ids
            .iter()
            .zip(sum_col.values.iter())
            .filter(|&(id, _)| !sum_col.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
            .collect();

        let mut groups: HashMap<i64, f64> = HashMap::new();

        for (record_id, group_value) in group_col.scan() {
            let key = group_value as i64;
            if let Some(&sum_value) = sum_values.get(&record_id) {
                *groups.entry(key).or_insert(0.0) += sum_value;
            }
        }

        Some(groups)
    }

    /// GROUP BY with COUNT aggregation
    ///
    /// Returns HashMap<group_key, count>
    pub fn group_by_count(&self, group_field: &str) -> Option<HashMap<i64, usize>> {
        let group_col = self.columns.get(group_field)?;

        let mut groups: HashMap<i64, usize> = HashMap::new();

        for (_, group_value) in group_col.scan() {
            let key = group_value as i64;
            *groups.entry(key).or_insert(0) += 1;
        }

        Some(groups)
    }

    /// GROUP BY with AVG aggregation
    ///
    /// Returns HashMap<group_key, avg_value>
    pub fn group_by_avg(&self, group_field: &str, avg_field: &str) -> Option<HashMap<i64, f64>> {
        let group_col = self.columns.get(group_field)?;
        let avg_col = self.columns.get(avg_field)?;

        // Build map from record_id to avg value
        let avg_values: HashMap<RecordId, f64> = avg_col
            .record_ids
            .iter()
            .zip(avg_col.values.iter())
            .filter(|&(id, _)| !avg_col.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
            .collect();

        let mut sums: HashMap<i64, f64> = HashMap::new();
        let mut counts: HashMap<i64, usize> = HashMap::new();

        for (record_id, group_value) in group_col.scan() {
            let key = group_value as i64;
            if let Some(&val) = avg_values.get(&record_id) {
                *sums.entry(key).or_insert(0.0) += val;
                *counts.entry(key).or_insert(0) += 1;
            }
        }

        let mut result: HashMap<i64, f64> = HashMap::new();
        for (key, sum) in sums {
            if let Some(&count) = counts.get(&key) {
                if count > 0 {
                    result.insert(key, sum / count as f64);
                }
            }
        }

        Some(result)
    }

    /// GROUP BY with MIN aggregation
    pub fn group_by_min(&self, group_field: &str, min_field: &str) -> Option<HashMap<i64, f64>> {
        let group_col = self.columns.get(group_field)?;
        let min_col = self.columns.get(min_field)?;

        let min_values: HashMap<RecordId, f64> = min_col
            .record_ids
            .iter()
            .zip(min_col.values.iter())
            .filter(|&(id, _)| !min_col.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
            .collect();

        let mut groups: HashMap<i64, f64> = HashMap::new();

        for (record_id, group_value) in group_col.scan() {
            let key = group_value as i64;
            if let Some(&val) = min_values.get(&record_id) {
                groups
                    .entry(key)
                    .and_modify(|e| *e = e.min(val))
                    .or_insert(val);
            }
        }

        Some(groups)
    }

    /// GROUP BY with MAX aggregation
    pub fn group_by_max(&self, group_field: &str, max_field: &str) -> Option<HashMap<i64, f64>> {
        let group_col = self.columns.get(group_field)?;
        let max_col = self.columns.get(max_field)?;

        let max_values: HashMap<RecordId, f64> = max_col
            .record_ids
            .iter()
            .zip(max_col.values.iter())
            .filter(|&(id, _)| !max_col.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
            .collect();

        let mut groups: HashMap<i64, f64> = HashMap::new();

        for (record_id, group_value) in group_col.scan() {
            let key = group_value as i64;
            if let Some(&val) = max_values.get(&record_id) {
                groups
                    .entry(key)
                    .and_modify(|e| *e = e.max(val))
                    .or_insert(val);
            }
        }

        Some(groups)
    }

    /// GROUP BY with multiple aggregations
    ///
    /// Returns HashMap<group_key, GroupAggregates> with sum, count, min, max
    pub fn group_by_multi(
        &self,
        group_field: &str,
        agg_field: &str,
    ) -> Option<HashMap<i64, GroupAggregates>> {
        let group_col = self.columns.get(group_field)?;
        let agg_col = self.columns.get(agg_field)?;

        let agg_values: HashMap<RecordId, f64> = agg_col
            .record_ids
            .iter()
            .zip(agg_col.values.iter())
            .filter(|&(id, _)| !agg_col.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
            .collect();

        let mut groups: HashMap<i64, GroupAggregates> = HashMap::new();

        for (record_id, group_value) in group_col.scan() {
            let key = group_value as i64;
            if let Some(&val) = agg_values.get(&record_id) {
                groups
                    .entry(key)
                    .and_modify(|g| g.add(val))
                    .or_insert_with(|| GroupAggregates::new(val));
            }
        }

        Some(groups)
    }

    // =========================================================================
    // Window Functions
    // =========================================================================

    /// Compute ROW_NUMBER() window function
    ///
    /// Returns values indexed by record_id in the order they appear after sorting.
    /// If partition_col is Some, numbering restarts for each partition.
    pub fn compute_row_number(
        &self,
        order_col: &str,
        order_asc: bool,
        partition_col: Option<&str>,
    ) -> HashMap<RecordId, f64> {
        self.compute_window_generic(order_col, order_asc, partition_col, |partition_values| {
            partition_values
                .iter()
                .enumerate()
                .map(|(i, &(record_id, _))| (record_id, (i + 1) as f64))
                .collect()
        })
    }

    /// Compute RANK() window function
    ///
    /// Ranks with gaps for ties (1, 1, 3, 4 for ties at position 1).
    pub fn compute_rank(
        &self,
        order_col: &str,
        order_asc: bool,
        partition_col: Option<&str>,
    ) -> HashMap<RecordId, f64> {
        self.compute_window_generic(order_col, order_asc, partition_col, |partition_values| {
            let mut result = HashMap::new();
            let mut current_rank = 1;
            let mut prev_value: Option<f64> = None;

            for (i, &(record_id, value)) in partition_values.iter().enumerate() {
                if let Some(prev) = prev_value {
                    if (value - prev).abs() >= 1e-10 {
                        // Different value - update rank (skip for ties)
                        current_rank = i + 1;
                    }
                }
                result.insert(record_id, current_rank as f64);
                prev_value = Some(value);
            }
            result
        })
    }

    /// Compute DENSE_RANK() window function
    ///
    /// Ranks without gaps for ties (1, 1, 2, 3 for ties at position 1).
    pub fn compute_dense_rank(
        &self,
        order_col: &str,
        order_asc: bool,
        partition_col: Option<&str>,
    ) -> HashMap<RecordId, f64> {
        self.compute_window_generic(order_col, order_asc, partition_col, |partition_values| {
            let mut result = HashMap::new();
            let mut current_rank = 1;
            let mut prev_value: Option<f64> = None;

            for &(record_id, value) in partition_values.iter() {
                if let Some(prev) = prev_value {
                    if (value - prev).abs() >= 1e-10 {
                        // Different value - increment rank
                        current_rank += 1;
                    }
                }
                result.insert(record_id, current_rank as f64);
                prev_value = Some(value);
            }
            result
        })
    }

    /// Compute NTILE(n) window function
    ///
    /// Divides rows into n roughly equal buckets.
    pub fn compute_ntile(
        &self,
        n: usize,
        order_col: &str,
        order_asc: bool,
        partition_col: Option<&str>,
    ) -> HashMap<RecordId, f64> {
        self.compute_window_generic(order_col, order_asc, partition_col, |partition_values| {
            let len = partition_values.len();
            let bucket_size = len / n;
            let remainder = len % n;

            let mut result = HashMap::new();
            let mut current_bucket = 1;
            let mut count_in_bucket = 0;
            let current_bucket_size = |b: usize| -> usize {
                if b <= remainder {
                    bucket_size + 1
                } else {
                    bucket_size
                }
            };

            for &(record_id, _) in partition_values.iter() {
                result.insert(record_id, current_bucket as f64);
                count_in_bucket += 1;
                if count_in_bucket >= current_bucket_size(current_bucket) && current_bucket < n {
                    current_bucket += 1;
                    count_in_bucket = 0;
                }
            }
            result
        })
    }

    /// Compute LEAD(col, offset) window function
    ///
    /// Returns the value of a column from a subsequent row.
    pub fn compute_lead(
        &self,
        value_col: &str,
        offset: usize,
        order_col: &str,
        order_asc: bool,
        partition_col: Option<&str>,
    ) -> HashMap<RecordId, f64> {
        let value_column = match self.columns.get(value_col) {
            Some(c) => c,
            None => return HashMap::new(),
        };
        let value_map: HashMap<RecordId, f64> = value_column
            .record_ids
            .iter()
            .zip(value_column.values.iter())
            .filter(|&(id, _)| !value_column.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
            .collect();

        self.compute_window_generic(order_col, order_asc, partition_col, |partition_values| {
            let mut result = HashMap::new();
            let len = partition_values.len();

            for (i, &(record_id, _)) in partition_values.iter().enumerate() {
                let lead_idx = i + offset;
                let lead_value = if lead_idx < len {
                    let (lead_record_id, _) = partition_values[lead_idx];
                    value_map.get(&lead_record_id).copied().unwrap_or(f64::NAN)
                } else {
                    f64::NAN
                };
                result.insert(record_id, lead_value);
            }
            result
        })
    }

    /// Compute LAG(col, offset) window function
    ///
    /// Returns the value of a column from a preceding row.
    pub fn compute_lag(
        &self,
        value_col: &str,
        offset: usize,
        order_col: &str,
        order_asc: bool,
        partition_col: Option<&str>,
    ) -> HashMap<RecordId, f64> {
        let value_column = match self.columns.get(value_col) {
            Some(c) => c,
            None => return HashMap::new(),
        };
        let value_map: HashMap<RecordId, f64> = value_column
            .record_ids
            .iter()
            .zip(value_column.values.iter())
            .filter(|&(id, _)| !value_column.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
            .collect();

        self.compute_window_generic(order_col, order_asc, partition_col, |partition_values| {
            let mut result = HashMap::new();

            for (i, &(record_id, _)) in partition_values.iter().enumerate() {
                let lag_value = if i >= offset {
                    let (lag_record_id, _) = partition_values[i - offset];
                    value_map.get(&lag_record_id).copied().unwrap_or(f64::NAN)
                } else {
                    f64::NAN
                };
                result.insert(record_id, lag_value);
            }
            result
        })
    }

    /// Compute FIRST_VALUE(col) window function
    pub fn compute_first_value(
        &self,
        value_col: &str,
        order_col: &str,
        order_asc: bool,
        partition_col: Option<&str>,
    ) -> HashMap<RecordId, f64> {
        let value_column = match self.columns.get(value_col) {
            Some(c) => c,
            None => return HashMap::new(),
        };
        let value_map: HashMap<RecordId, f64> = value_column
            .record_ids
            .iter()
            .zip(value_column.values.iter())
            .filter(|&(id, _)| !value_column.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
            .collect();

        self.compute_window_generic(order_col, order_asc, partition_col, |partition_values| {
            let mut result = HashMap::new();
            if partition_values.is_empty() {
                return result;
            }

            let (first_record_id, _) = partition_values[0];
            let first_value = value_map.get(&first_record_id).copied().unwrap_or(f64::NAN);

            for &(record_id, _) in partition_values.iter() {
                result.insert(record_id, first_value);
            }
            result
        })
    }

    /// Compute LAST_VALUE(col) window function
    pub fn compute_last_value(
        &self,
        value_col: &str,
        order_col: &str,
        order_asc: bool,
        partition_col: Option<&str>,
    ) -> HashMap<RecordId, f64> {
        let value_column = match self.columns.get(value_col) {
            Some(c) => c,
            None => return HashMap::new(),
        };
        let value_map: HashMap<RecordId, f64> = value_column
            .record_ids
            .iter()
            .zip(value_column.values.iter())
            .filter(|&(id, _)| !value_column.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
            .collect();

        self.compute_window_generic(order_col, order_asc, partition_col, |partition_values| {
            let mut result = HashMap::new();
            if partition_values.is_empty() {
                return result;
            }

            let (last_record_id, _) = partition_values[partition_values.len() - 1];
            let last_value = value_map.get(&last_record_id).copied().unwrap_or(f64::NAN);

            for &(record_id, _) in partition_values.iter() {
                result.insert(record_id, last_value);
            }
            result
        })
    }

    /// Compute running SUM() window function
    pub fn compute_running_sum(
        &self,
        value_col: &str,
        order_col: &str,
        order_asc: bool,
        partition_col: Option<&str>,
    ) -> HashMap<RecordId, f64> {
        let value_column = match self.columns.get(value_col) {
            Some(c) => c,
            None => return HashMap::new(),
        };
        let value_map: HashMap<RecordId, f64> = value_column
            .record_ids
            .iter()
            .zip(value_column.values.iter())
            .filter(|&(id, _)| !value_column.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
            .collect();

        self.compute_window_generic(order_col, order_asc, partition_col, |partition_values| {
            let mut result = HashMap::new();
            let mut running_sum = 0.0;

            for &(record_id, _) in partition_values.iter() {
                let value = value_map.get(&record_id).copied().unwrap_or(0.0);
                running_sum += value;
                result.insert(record_id, running_sum);
            }
            result
        })
    }

    /// Compute running AVG() window function
    pub fn compute_running_avg(
        &self,
        value_col: &str,
        order_col: &str,
        order_asc: bool,
        partition_col: Option<&str>,
    ) -> HashMap<RecordId, f64> {
        let value_column = match self.columns.get(value_col) {
            Some(c) => c,
            None => return HashMap::new(),
        };
        let value_map: HashMap<RecordId, f64> = value_column
            .record_ids
            .iter()
            .zip(value_column.values.iter())
            .filter(|&(id, _)| !value_column.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
            .collect();

        self.compute_window_generic(order_col, order_asc, partition_col, |partition_values| {
            let mut result = HashMap::new();
            let mut running_sum = 0.0;
            let mut count = 0;

            for &(record_id, _) in partition_values.iter() {
                let value = value_map.get(&record_id).copied().unwrap_or(0.0);
                running_sum += value;
                count += 1;
                result.insert(record_id, running_sum / count as f64);
            }
            result
        })
    }

    /// Compute running COUNT() window function
    pub fn compute_running_count(
        &self,
        order_col: &str,
        order_asc: bool,
        partition_col: Option<&str>,
    ) -> HashMap<RecordId, f64> {
        self.compute_window_generic(order_col, order_asc, partition_col, |partition_values| {
            let mut result = HashMap::new();
            for (i, &(record_id, _)) in partition_values.iter().enumerate() {
                result.insert(record_id, (i + 1) as f64);
            }
            result
        })
    }

    /// Generic helper for computing window functions
    fn compute_window_generic<F>(
        &self,
        order_col: &str,
        order_asc: bool,
        partition_col: Option<&str>,
        compute_fn: F,
    ) -> HashMap<RecordId, f64>
    where
        F: Fn(&[(RecordId, f64)]) -> HashMap<RecordId, f64>,
    {
        let order_column = match self.columns.get(order_col) {
            Some(c) => c,
            None => return HashMap::new(),
        };

        // Collect (record_id, order_value) pairs
        let mut values: Vec<(RecordId, f64)> = order_column
            .record_ids
            .iter()
            .zip(order_column.values.iter())
            .filter(|&(id, _)| !order_column.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v))
            .collect();

        // If no partition, treat all rows as one partition
        if partition_col.is_none() {
            // Sort by order column
            if order_asc {
                values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            return compute_fn(&values);
        }

        // With partition: group by partition value, then compute for each partition
        let partition_col_name = partition_col.unwrap();
        let partition_column = match self.columns.get(partition_col_name) {
            Some(c) => c,
            None => return HashMap::new(),
        };

        // Build partition map
        let partition_map: HashMap<RecordId, i64> = partition_column
            .record_ids
            .iter()
            .zip(partition_column.values.iter())
            .filter(|&(id, _)| !partition_column.tombstones.contains(&id))
            .map(|(&id, &v)| (id, v as i64))
            .collect();

        // Group values by partition
        let mut partitions: HashMap<i64, Vec<(RecordId, f64)>> = HashMap::new();
        for (record_id, order_value) in values {
            if let Some(&partition_key) = partition_map.get(&record_id) {
                partitions
                    .entry(partition_key)
                    .or_default()
                    .push((record_id, order_value));
            }
        }

        // Compute for each partition
        let mut result = HashMap::new();
        for (_, mut partition_values) in partitions {
            // Sort partition by order column
            if order_asc {
                partition_values
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                partition_values
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }

            let partition_result = compute_fn(&partition_values);
            result.extend(partition_result);
        }

        result
    }

    /// Mark a record as deleted across all columns
    pub fn mark_deleted(&mut self, record_id: RecordId) {
        for column in self.columns.values_mut() {
            column.mark_deleted(record_id);
        }
    }

    /// Compact all columns (remove tombstoned entries)
    pub fn compact(&mut self) {
        for column in self.columns.values_mut() {
            column.compact();
        }
    }

    /// Check if any column needs compaction
    pub fn needs_compaction(&self) -> bool {
        self.columns.values().any(|c| c.needs_compaction())
    }
}

/// Result of a hash join operation
#[derive(Debug, Clone)]
pub struct JoinResult {
    /// Matched pairs: (build_record_id, probe_record_id)
    pub matches: Vec<(RecordId, RecordId)>,
    /// Name of the build column
    pub build_field: String,
    /// Name of the probe column
    pub probe_field: String,
}

impl JoinResult {
    /// Number of matched pairs
    pub fn len(&self) -> usize {
        self.matches.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    /// Get all build-side record IDs from the join
    pub fn build_record_ids(&self) -> Vec<RecordId> {
        self.matches.iter().map(|(b, _)| *b).collect()
    }

    /// Get all probe-side record IDs from the join
    pub fn probe_record_ids(&self) -> Vec<RecordId> {
        self.matches.iter().map(|(_, p)| *p).collect()
    }
}

/// Aggregates for a single group in GROUP BY operations
#[derive(Debug, Clone)]
pub struct GroupAggregates {
    /// Sum of values in group
    pub sum: f64,
    /// Count of values in group
    pub count: usize,
    /// Minimum value in group
    pub min: f64,
    /// Maximum value in group
    pub max: f64,
}

impl GroupAggregates {
    /// Create new aggregates from first value
    pub fn new(value: f64) -> Self {
        Self {
            sum: value,
            count: 1,
            min: value,
            max: value,
        }
    }

    /// Add a value to the aggregates
    pub fn add(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Get average
    pub fn avg(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

impl Default for GroupAggregates {
    fn default() -> Self {
        Self {
            sum: 0.0,
            count: 0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }
}

/// Statistics for all columns
#[derive(Debug, Clone)]
pub struct ColumnarStats {
    pub num_columns: usize,
    pub total_values: usize,
    pub columns: Vec<ColumnStats>,
}

/// Statistics for a single column
#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub name: String,
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub sum: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_basic() {
        let mut col = Column::new();
        col.push(1, 10.0);
        col.push(2, 20.0);
        col.push(3, 30.0);

        assert_eq!(col.len(), 3);
        assert_eq!(col.sum(), 60.0);
        assert_eq!(col.avg(), 20.0);
        assert_eq!(col.min(), 10.0);
        assert_eq!(col.max(), 30.0);
    }

    #[test]
    fn test_column_range() {
        let mut col = Column::new();
        for i in 0..100 {
            col.push(i as RecordId, i as f64);
        }

        assert_eq!(col.count_range(10.0, 20.0), 10);
        assert_eq!(col.sum_range(10.0, 20.0), 145.0); // 10+11+...+19
    }

    #[test]
    fn test_columnar_store() {
        let mut store = ColumnarStore::new();

        // Simulate ingesting documents
        store.record_value("price", 1, &Value::Float(100.0));
        store.record_value("price", 2, &Value::Float(200.0));
        store.record_value("price", 3, &Value::Float(300.0));
        store.record_value("quantity", 1, &Value::Int(10));
        store.record_value("quantity", 2, &Value::Int(20));
        store.record_value("quantity", 3, &Value::Int(30));

        assert_eq!(store.sum("price"), Some(600.0));
        assert_eq!(store.count("price"), Some(3));
        assert_eq!(store.avg("quantity"), Some(20.0));
    }

    #[test]
    fn test_tombstones() {
        let mut col = Column::new();
        col.push(1, 10.0);
        col.push(2, 20.0);
        col.push(3, 30.0);

        assert_eq!(col.sum(), 60.0);
        assert_eq!(col.count(), 3);

        // Mark record 2 as deleted
        col.mark_deleted(2);

        // Raw stats unchanged (lazy deletion)
        assert_eq!(col.sum(), 60.0);
        assert_eq!(col.count(), 3);

        // Live stats account for tombstone
        assert_eq!(col.sum_live(), 40.0); // 10 + 30
        assert_eq!(col.count_live(), 2);
        assert_eq!(col.avg_live(), 20.0); // 40 / 2

        // is_deleted works
        assert!(!col.is_deleted(1));
        assert!(col.is_deleted(2));
        assert!(!col.is_deleted(3));
    }

    #[test]
    fn test_compaction() {
        let mut col = Column::new();
        col.push(1, 10.0);
        col.push(2, 20.0);
        col.push(3, 30.0);
        col.push(4, 40.0);
        col.push(5, 50.0);

        // Delete 2 out of 5 (40% - exceeds 10% threshold)
        col.mark_deleted(2);
        col.mark_deleted(4);

        assert!(col.needs_compaction());
        assert_eq!(col.tombstone_count(), 2);

        // Compact
        col.compact();

        // After compaction
        assert!(!col.needs_compaction());
        assert_eq!(col.tombstone_count(), 0);
        assert_eq!(col.count(), 3);
        assert_eq!(col.sum(), 90.0); // 10 + 30 + 50
        assert_eq!(col.min(), 10.0);
        assert_eq!(col.max(), 50.0);
    }

    #[test]
    fn test_hash_index() {
        let mut col = Column::new();
        col.push(100, 1.0); // key 1
        col.push(101, 2.0); // key 2
        col.push(102, 1.0); // key 1 (duplicate)
        col.push(103, 3.0); // key 3

        let index = col.build_hash_index();

        assert_eq!(index.get(&1).map(|v| v.len()), Some(2)); // Two records with key 1
        assert_eq!(index.get(&2).map(|v| v.len()), Some(1));
        assert_eq!(index.get(&3).map(|v| v.len()), Some(1));
        assert!(index.get(&4).is_none()); // No key 4
    }

    #[test]
    fn test_hash_join() {
        let mut store = ColumnarStore::new();

        // Orders table: o_orderkey
        store.record_value("o_orderkey", 1, &Value::Int(100));
        store.record_value("o_orderkey", 2, &Value::Int(200));
        store.record_value("o_orderkey", 3, &Value::Int(300));

        // Lineitem table: l_orderkey (foreign key to orders)
        store.record_value("l_orderkey", 10, &Value::Int(100)); // matches order 1
        store.record_value("l_orderkey", 11, &Value::Int(100)); // matches order 1
        store.record_value("l_orderkey", 12, &Value::Int(200)); // matches order 2
        store.record_value("l_orderkey", 13, &Value::Int(400)); // no match

        // Join: SELECT * FROM orders JOIN lineitem ON o_orderkey = l_orderkey
        let result = store.hash_join("o_orderkey", "l_orderkey").unwrap();

        assert_eq!(result.len(), 3); // 3 matches
        assert!(result.matches.contains(&(1, 10))); // order 1 -> lineitem 10
        assert!(result.matches.contains(&(1, 11))); // order 1 -> lineitem 11
        assert!(result.matches.contains(&(2, 12))); // order 2 -> lineitem 12
    }

    #[test]
    fn test_hash_join_with_sum() {
        let mut store = ColumnarStore::new();

        // Orders: just the join key
        store.record_value("o_orderkey", 1, &Value::Int(100));
        store.record_value("o_orderkey", 2, &Value::Int(200));

        // Lineitem: join key + price
        store.record_value("l_orderkey", 10, &Value::Int(100));
        store.record_value("l_extendedprice", 10, &Value::Float(1000.0));

        store.record_value("l_orderkey", 11, &Value::Int(100));
        store.record_value("l_extendedprice", 11, &Value::Float(500.0));

        store.record_value("l_orderkey", 12, &Value::Int(200));
        store.record_value("l_extendedprice", 12, &Value::Float(750.0));

        store.record_value("l_orderkey", 13, &Value::Int(400)); // no match
        store.record_value("l_extendedprice", 13, &Value::Float(999.0));

        // SUM(l_extendedprice) for matching lineitems
        let sum = store
            .hash_join_sum("o_orderkey", "l_orderkey", "l_extendedprice")
            .unwrap();

        // Should be 1000 + 500 + 750 = 2250 (excluding 999 which has no matching order)
        assert_eq!(sum, 2250.0);
    }

    #[test]
    fn test_store_mark_deleted() {
        let mut store = ColumnarStore::new();

        store.record_value("price", 1, &Value::Float(100.0));
        store.record_value("price", 2, &Value::Float(200.0));
        store.record_value("quantity", 1, &Value::Int(10));
        store.record_value("quantity", 2, &Value::Int(20));

        // Delete record 1 across all columns
        store.mark_deleted(1);

        let price_col = store.get_column("price").unwrap();
        let qty_col = store.get_column("quantity").unwrap();

        assert!(price_col.is_deleted(1));
        assert!(qty_col.is_deleted(1));
        assert!(!price_col.is_deleted(2));
        assert!(!qty_col.is_deleted(2));
    }

    #[test]
    fn test_left_outer_join() {
        let mut store = ColumnarStore::new();

        // Orders: 3 orders
        store.record_value("o_orderkey", 1, &Value::Int(100));
        store.record_value("o_orderkey", 2, &Value::Int(200));
        store.record_value("o_orderkey", 3, &Value::Int(300));

        // Lineitem: only matches orders 100 and 200
        store.record_value("l_orderkey", 10, &Value::Int(100));
        store.record_value("l_orderkey", 11, &Value::Int(200));

        let result = store.left_outer_join("o_orderkey", "l_orderkey").unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.contains(&(1, Some(10)))); // order 100 matches lineitem 10
        assert!(result.contains(&(2, Some(11)))); // order 200 matches lineitem 11
        assert!(result.contains(&(3, None))); // order 300 has no match
    }

    #[test]
    fn test_semi_join() {
        let mut store = ColumnarStore::new();

        // Build side: orders
        store.record_value("o_orderkey", 1, &Value::Int(100));
        store.record_value("o_orderkey", 2, &Value::Int(200));
        store.record_value("o_orderkey", 3, &Value::Int(300));

        // Probe side: lineitem (only has 100 and 200)
        store.record_value("l_orderkey", 10, &Value::Int(100));
        store.record_value("l_orderkey", 11, &Value::Int(200));

        let result = store.semi_join("o_orderkey", "l_orderkey").unwrap();

        // Semi-join returns build records that have matches
        assert_eq!(result.len(), 2);
        assert!(result.contains(&1)); // order 100 has match
        assert!(result.contains(&2)); // order 200 has match
        assert!(!result.contains(&3)); // order 300 has no match
    }

    #[test]
    fn test_anti_join() {
        let mut store = ColumnarStore::new();

        // Build side: orders
        store.record_value("o_orderkey", 1, &Value::Int(100));
        store.record_value("o_orderkey", 2, &Value::Int(200));
        store.record_value("o_orderkey", 3, &Value::Int(300));

        // Probe side: lineitem (only has 100 and 200)
        store.record_value("l_orderkey", 10, &Value::Int(100));
        store.record_value("l_orderkey", 11, &Value::Int(200));

        let result = store.anti_join("o_orderkey", "l_orderkey").unwrap();

        // Anti-join returns build records WITHOUT matches
        assert_eq!(result.len(), 1);
        assert!(!result.contains(&1)); // order 100 has match
        assert!(!result.contains(&2)); // order 200 has match
        assert!(result.contains(&3)); // order 300 has no match - returned
    }

    #[test]
    fn test_group_by_sum() {
        let mut store = ColumnarStore::new();

        // Group field: category
        store.record_value("category", 1, &Value::Int(1)); // category 1
        store.record_value("category", 2, &Value::Int(1)); // category 1
        store.record_value("category", 3, &Value::Int(2)); // category 2
        store.record_value("category", 4, &Value::Int(2)); // category 2
        store.record_value("category", 5, &Value::Int(2)); // category 2

        // Sum field: price
        store.record_value("price", 1, &Value::Float(100.0));
        store.record_value("price", 2, &Value::Float(150.0));
        store.record_value("price", 3, &Value::Float(200.0));
        store.record_value("price", 4, &Value::Float(250.0));
        store.record_value("price", 5, &Value::Float(300.0));

        let result = store.group_by_sum("category", "price").unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result.get(&1), Some(&250.0)); // 100 + 150
        assert_eq!(result.get(&2), Some(&750.0)); // 200 + 250 + 300
    }

    #[test]
    fn test_group_by_count() {
        let mut store = ColumnarStore::new();

        store.record_value("status", 1, &Value::Int(1)); // status 1
        store.record_value("status", 2, &Value::Int(1)); // status 1
        store.record_value("status", 3, &Value::Int(2)); // status 2
        store.record_value("status", 4, &Value::Int(3)); // status 3
        store.record_value("status", 5, &Value::Int(3)); // status 3
        store.record_value("status", 6, &Value::Int(3)); // status 3

        let result = store.group_by_count("status").unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result.get(&1), Some(&2)); // 2 records with status 1
        assert_eq!(result.get(&2), Some(&1)); // 1 record with status 2
        assert_eq!(result.get(&3), Some(&3)); // 3 records with status 3
    }

    #[test]
    fn test_group_by_avg() {
        let mut store = ColumnarStore::new();

        store.record_value("group", 1, &Value::Int(1));
        store.record_value("group", 2, &Value::Int(1));
        store.record_value("group", 3, &Value::Int(2));

        store.record_value("value", 1, &Value::Float(10.0));
        store.record_value("value", 2, &Value::Float(20.0));
        store.record_value("value", 3, &Value::Float(30.0));

        let result = store.group_by_avg("group", "value").unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result.get(&1), Some(&15.0)); // (10 + 20) / 2
        assert_eq!(result.get(&2), Some(&30.0)); // 30 / 1
    }

    #[test]
    fn test_group_by_multi() {
        let mut store = ColumnarStore::new();

        store.record_value("category", 1, &Value::Int(1));
        store.record_value("category", 2, &Value::Int(1));
        store.record_value("category", 3, &Value::Int(2));

        store.record_value("amount", 1, &Value::Float(100.0));
        store.record_value("amount", 2, &Value::Float(200.0));
        store.record_value("amount", 3, &Value::Float(50.0));

        let result = store.group_by_multi("category", "amount").unwrap();

        let cat1 = result.get(&1).unwrap();
        assert_eq!(cat1.sum, 300.0); // 100 + 200
        assert_eq!(cat1.count, 2);
        assert_eq!(cat1.min, 100.0);
        assert_eq!(cat1.max, 200.0);
        assert_eq!(cat1.avg(), 150.0); // 300 / 2

        let cat2 = result.get(&2).unwrap();
        assert_eq!(cat2.sum, 50.0);
        assert_eq!(cat2.count, 1);
        assert_eq!(cat2.min, 50.0);
        assert_eq!(cat2.max, 50.0);
    }
}
