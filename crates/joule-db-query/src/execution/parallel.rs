//! Parallel Query Execution
//!
//! Implements parallel execution of query operations across multiple CPU cores
//! for improved performance on multi-core systems.
//!
//! ## Design
//!
//! - Work-stealing scheduler for load balancing
//! - Parallel table scans with partition-based parallelism
//! - Parallel joins (hash join, sort-merge join)
//! - Parallel aggregations with merge phase
//! - Automatic parallelism detection based on data size

use crate::error::QueryResult;
use crate::execution::{ResultSet, Row};
use crate::timeout::CheckpointContext;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for parallel execution
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum number of threads to use
    pub max_threads: usize,
    /// Minimum rows per partition for parallel scan
    pub min_rows_per_partition: usize,
    /// Enable parallel execution
    pub enabled: bool,
    /// Threshold for enabling parallelism (number of rows)
    pub parallelism_threshold: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_threads: num_cpus::get(),
            min_rows_per_partition: 1000,
            enabled: true,
            parallelism_threshold: 10000,
        }
    }
}

impl ParallelConfig {
    /// Create with custom thread count
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.max_threads = threads;
        self
    }

    /// Disable parallel execution
    pub fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Set parallelism threshold
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.parallelism_threshold = threshold;
        self
    }
}

/// Parallel query executor
pub struct ParallelExecutor {
    config: ParallelConfig,
}

impl ParallelExecutor {
    /// Create new parallel executor
    pub fn new(config: ParallelConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default() -> Self {
        Self::new(ParallelConfig::default())
    }

    /// Execute a scan operation in parallel
    pub fn parallel_scan<F>(
        &self,
        total_rows: usize,
        scan_fn: F,
        checkpoint_ctx: Option<&CheckpointContext>,
    ) -> QueryResult<Vec<Row>>
    where
        F: Fn(usize, usize) -> QueryResult<Vec<Row>> + Send + Sync,
    {
        if !self.config.enabled || total_rows < self.config.parallelism_threshold {
            // Sequential execution for small datasets
            return scan_fn(0, total_rows);
        }

        // Determine number of partitions
        let num_partitions = (total_rows / self.config.min_rows_per_partition)
            .min(self.config.max_threads)
            .max(1);

        let rows_per_partition = (total_rows + num_partitions - 1) / num_partitions;

        // Execute partitions in parallel
        #[cfg(feature = "parallel")]
        let results: QueryResult<Vec<_>> = (0..num_partitions)
            .into_par_iter()
            .map(|i| {
                // Check cancellation if checkpoint context provided
                if let Some(ctx) = checkpoint_ctx {
                    ctx.checkpoint()?;
                }

                let start = i * rows_per_partition;
                let end = (start + rows_per_partition).min(total_rows);
                scan_fn(start, end)
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: QueryResult<Vec<_>> = (0..num_partitions)
            .map(|i| {
                // Check cancellation if checkpoint context provided
                if let Some(ctx) = checkpoint_ctx {
                    ctx.checkpoint()?;
                }

                let start = i * rows_per_partition;
                let end = (start + rows_per_partition).min(total_rows);
                scan_fn(start, end)
            })
            .collect();

        // Flatten results
        Ok(results?.into_iter().flatten().collect())
    }

    /// Execute a filter operation in parallel
    pub fn parallel_filter(
        &self,
        rows: Vec<Row>,
        filter_fn: impl Fn(&Row) -> bool + Send + Sync,
        checkpoint_ctx: Option<&CheckpointContext>,
    ) -> QueryResult<Vec<Row>> {
        if !self.config.enabled || rows.len() < self.config.parallelism_threshold {
            // Sequential execution
            return Ok(rows.into_iter().filter(|r| filter_fn(r)).collect());
        }

        // Parallel filter
        #[cfg(feature = "parallel")]
        let filtered: Vec<Row> = rows
            .into_par_iter()
            .filter(|row| {
                // Check cancellation periodically
                if let Some(ctx) = checkpoint_ctx {
                    if let Err(_) = ctx.checkpoint_throttled() {
                        return false;
                    }
                }
                filter_fn(row)
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let filtered: Vec<Row> = rows
            .into_iter()
            .filter(|row| {
                // Check cancellation periodically
                if let Some(ctx) = checkpoint_ctx {
                    if let Err(_) = ctx.checkpoint_throttled() {
                        return false;
                    }
                }
                filter_fn(row)
            })
            .collect();

        Ok(filtered)
    }

    /// Execute a join operation in parallel (hash join)
    pub fn parallel_hash_join(
        &self,
        left: ResultSet,
        right: ResultSet,
        join_key_fn: impl Fn(&Row) -> u64 + Send + Sync,
        join_condition: impl Fn(&Row, &Row) -> bool + Send + Sync,
        checkpoint_ctx: Option<&CheckpointContext>,
    ) -> QueryResult<ResultSet> {
        if !self.config.enabled
            || (left.rows.len() + right.rows.len()) < self.config.parallelism_threshold
        {
            // Sequential join for small datasets
            return self.sequential_hash_join(left, right, join_key_fn, join_condition);
        }

        // Build hash table from right side in parallel
        #[cfg(feature = "parallel")]
        let right_hash: std::collections::HashMap<u64, Vec<Row>> = right
            .rows
            .into_par_iter()
            .fold(
                || std::collections::HashMap::new(),
                |mut map, row| {
                    let key = join_key_fn(&row);
                    map.entry(key).or_insert_with(Vec::new).push(row);
                    map
                },
            )
            .reduce(
                || std::collections::HashMap::new(),
                |mut a, b| {
                    for (k, v) in b {
                        a.entry(k).or_insert_with(Vec::new).extend(v);
                    }
                    a
                },
            );

        #[cfg(not(feature = "parallel"))]
        let right_hash: std::collections::HashMap<u64, Vec<Row>> = {
            let mut map = std::collections::HashMap::new();
            for row in right.rows {
                let key = join_key_fn(&row);
                map.entry(key).or_insert_with(Vec::new).push(row);
            }
            map
        };

        // Probe left side in parallel
        #[cfg(feature = "parallel")]
        let joined_rows: QueryResult<Vec<Row>> = {
            let row_batches: QueryResult<Vec<Vec<Row>>> = left
                .rows
                .into_par_iter()
                .map(|left_row| {
                    // Check cancellation
                    if let Some(ctx) = checkpoint_ctx {
                        ctx.checkpoint_throttled()?;
                    }

                    let key = join_key_fn(&left_row);
                    if let Some(right_rows) = right_hash.get(&key) {
                        let mut results = Vec::new();
                        for right_row in right_rows {
                            if join_condition(&left_row, right_row) {
                                // Combine rows
                                let mut combined_values = left_row.values.clone();
                                combined_values.extend(right_row.values.clone());
                                results.push(Row::new(combined_values));
                            }
                        }
                        Ok(results)
                    } else {
                        Ok(Vec::new())
                    }
                })
                .collect();
            row_batches.map(|batches| batches.into_iter().flatten().collect())
        };

        #[cfg(not(feature = "parallel"))]
        let joined_rows: QueryResult<Vec<Row>> = left
            .rows
            .into_iter()
            .map(|left_row| {
                // Check cancellation
                if let Some(ctx) = checkpoint_ctx {
                    ctx.checkpoint_throttled()?;
                }

                let key = join_key_fn(&left_row);
                if let Some(right_rows) = right_hash.get(&key) {
                    let mut results = Vec::new();
                    for right_row in right_rows {
                        if join_condition(&left_row, right_row) {
                            // Combine rows
                            let mut combined_values = left_row.values.clone();
                            combined_values.extend(right_row.values.clone());
                            results.push(Row::new(combined_values));
                        }
                    }
                    Ok(results)
                } else {
                    Ok(Vec::new())
                }
            })
            .collect();

        // Flatten and create result set
        let mut combined_columns = left.columns.clone();
        combined_columns.extend(right.columns.clone());
        let mut result = ResultSet::with_columns(combined_columns);

        for row in joined_rows? {
            result.add_row(row);
        }

        Ok(result)
    }

    /// Sequential hash join (fallback)
    fn sequential_hash_join(
        &self,
        left: ResultSet,
        right: ResultSet,
        join_key_fn: impl Fn(&Row) -> u64,
        join_condition: impl Fn(&Row, &Row) -> bool,
    ) -> QueryResult<ResultSet> {
        // Build hash table from right
        let mut right_hash: std::collections::HashMap<u64, Vec<Row>> =
            std::collections::HashMap::new();
        for row in right.rows {
            let key = join_key_fn(&row);
            right_hash.entry(key).or_insert_with(Vec::new).push(row);
        }

        // Probe left
        let mut combined_columns = left.columns.clone();
        combined_columns.extend(right.columns.clone());
        let mut result = ResultSet::with_columns(combined_columns);

        for left_row in left.rows {
            let key = join_key_fn(&left_row);
            if let Some(right_rows) = right_hash.get(&key) {
                for right_row in right_rows {
                    if join_condition(&left_row, right_row) {
                        let mut combined_values = left_row.values.clone();
                        combined_values.extend(right_row.values.clone());
                        result.add_row(Row::new(combined_values));
                    }
                }
            }
        }

        Ok(result)
    }

    /// Execute aggregation in parallel
    pub fn parallel_aggregate<F, G>(
        &self,
        rows: Vec<Row>,
        group_key_fn: F,
        aggregate_fn: G,
        checkpoint_ctx: Option<&CheckpointContext>,
    ) -> QueryResult<Vec<Row>>
    where
        F: Fn(&Row) -> String + Send + Sync,
        G: Fn(&[Row]) -> QueryResult<Row> + Send + Sync,
    {
        if !self.config.enabled || rows.len() < self.config.parallelism_threshold {
            // Sequential aggregation
            let mut groups: std::collections::HashMap<String, Vec<Row>> =
                std::collections::HashMap::new();
            for row in rows {
                let key = group_key_fn(&row);
                groups.entry(key).or_insert_with(Vec::new).push(row);
            }

            let mut results = Vec::new();
            for group_rows in groups.values() {
                results.push(aggregate_fn(group_rows)?);
            }
            return Ok(results);
        }

        // Parallel aggregation: group in parallel, then aggregate
        #[cfg(feature = "parallel")]
        let groups: std::collections::HashMap<String, Vec<Row>> = rows
            .into_par_iter()
            .fold(
                || std::collections::HashMap::new(),
                |mut map, row| {
                    // Check cancellation
                    if let Some(ctx) = checkpoint_ctx {
                        if let Err(_) = ctx.checkpoint_throttled() {
                            return map;
                        }
                    }
                    let key = group_key_fn(&row);
                    map.entry(key).or_insert_with(Vec::new).push(row);
                    map
                },
            )
            .reduce(
                || std::collections::HashMap::new(),
                |mut a, b| {
                    for (k, mut v) in b {
                        a.entry(k).or_insert_with(Vec::new).append(&mut v);
                    }
                    a
                },
            );

        #[cfg(not(feature = "parallel"))]
        let groups: std::collections::HashMap<String, Vec<Row>> = {
            let mut map = std::collections::HashMap::new();
            for row in rows {
                // Check cancellation
                if let Some(ctx) = checkpoint_ctx {
                    if let Err(_) = ctx.checkpoint_throttled() {
                        continue;
                    }
                }
                let key = group_key_fn(&row);
                map.entry(key).or_insert_with(Vec::new).push(row);
            }
            map
        };

        // Aggregate each group (can also be parallelized)
        #[cfg(feature = "parallel")]
        let results: QueryResult<Vec<_>> = {
            let group_vec: Vec<&Vec<Row>> = groups.values().collect();
            group_vec
                .into_par_iter()
                .map(|group_rows| aggregate_fn(group_rows))
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let results: QueryResult<Vec<_>> = groups
            .values()
            .map(|group_rows| aggregate_fn(group_rows))
            .collect();

        Ok(results?)
    }

    /// Get configuration
    pub fn config(&self) -> &ParallelConfig {
        &self.config
    }
}

impl Default for ParallelExecutor {
    fn default() -> Self {
        Self::new(ParallelConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ResultSet;

    #[test]
    fn test_parallel_scan() {
        let executor = ParallelExecutor::default();

        let result = executor
            .parallel_scan(
                100,
                |start, end| {
                    Ok((start..end)
                        .map(|i| Row::new(vec![crate::ast::Value::Int(i as i64)]))
                        .collect())
                },
                None,
            )
            .unwrap();

        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_parallel_filter() {
        let executor = ParallelExecutor::default();

        let rows: Vec<Row> = (0..1000)
            .map(|i| Row::new(vec![crate::ast::Value::Int(i)]))
            .collect();

        let filtered = executor
            .parallel_filter(
                rows,
                |row| row.values[0].as_int().unwrap_or(0) % 2 == 0,
                None,
            )
            .unwrap();

        assert_eq!(filtered.len(), 500);
    }

    #[test]
    fn test_parallel_aggregate() {
        let executor = ParallelExecutor::default();

        let rows: Vec<Row> = (0..1000)
            .map(|i| {
                Row::new(vec![
                    crate::ast::Value::String(format!("group_{}", i % 10)),
                    crate::ast::Value::Int(i),
                ])
            })
            .collect();

        let aggregated = executor
            .parallel_aggregate(
                rows,
                |row| row.values[0].as_str().unwrap_or_default().to_string(),
                |group_rows| {
                    let sum: i64 = group_rows
                        .iter()
                        .map(|r| r.values[1].as_int().unwrap_or(0))
                        .sum();
                    Ok(Row::new(vec![
                        group_rows[0].values[0].clone(),
                        crate::ast::Value::Int(sum),
                    ]))
                },
                None,
            )
            .unwrap();

        assert_eq!(aggregated.len(), 10); // 10 groups
    }
}
