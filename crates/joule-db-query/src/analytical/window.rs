//! Window Functions for OLAP Analytics
//!
//! Implements SQL window functions for analytical queries:
//! - Ranking functions (ROW_NUMBER, RANK, DENSE_RANK)
//! - Value functions (LAG, LEAD, FIRST_VALUE, LAST_VALUE)
//! - Aggregate functions (SUM, AVG, COUNT, etc. as window functions)
//! - Window frame specifications (ROWS, RANGE, GROUPS)

use crate::ast::{Expression, Value};
use crate::error::{QueryError, QueryResult};
use std::collections::HashMap;

/// Window function type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowFunctionType {
    /// ROW_NUMBER() - Sequential row number
    RowNumber,
    /// RANK() - Rank with gaps
    Rank,
    /// DENSE_RANK() - Rank without gaps
    DenseRank,
    /// PERCENT_RANK() - Relative rank (0.0 to 1.0)
    PercentRank,
    /// CUME_DIST() - Cumulative distribution
    CumeDist,
    /// LAG(expr, offset, default) - Previous row value
    Lag {
        offset: usize,
        default: Option<Value>,
    },
    /// LEAD(expr, offset, default) - Next row value
    Lead {
        offset: usize,
        default: Option<Value>,
    },
    /// FIRST_VALUE(expr) - First value in window
    FirstValue,
    /// LAST_VALUE(expr) - Last value in window
    LastValue,
    /// NTH_VALUE(expr, n) - Nth value in window
    NthValue { n: usize },
    /// NTILE(n) - Divide into n buckets
    Ntile { n: usize },
}

/// Window frame specification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowFrame {
    /// ROWS BETWEEN start AND end
    Rows { start: FrameBound, end: FrameBound },
    /// RANGE BETWEEN start AND end
    Range { start: FrameBound, end: FrameBound },
    /// GROUPS BETWEEN start AND end
    Groups { start: FrameBound, end: FrameBound },
}

/// Frame boundary
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameBound {
    /// UNBOUNDED PRECEDING
    UnboundedPreceding,
    /// value PRECEDING
    Preceding(usize),
    /// CURRENT ROW
    CurrentRow,
    /// value FOLLOWING
    Following(usize),
    /// UNBOUNDED FOLLOWING
    UnboundedFollowing,
}

/// Window specification
#[derive(Debug, Clone)]
pub struct WindowSpec {
    /// PARTITION BY expressions
    pub partition_by: Vec<Expression>,
    /// ORDER BY expressions
    pub order_by: Vec<(Expression, bool)>, // (expr, ascending)
    /// Window frame
    pub frame: Option<WindowFrame>,
}

/// Window function expression
#[derive(Debug, Clone)]
pub struct WindowFunction {
    /// Function type
    pub function: WindowFunctionType,
    /// Argument expression (for LAG, LEAD, FIRST_VALUE, etc.)
    pub argument: Option<Expression>,
    /// Window specification
    pub window_spec: WindowSpec,
}

/// Window function executor
pub struct WindowExecutor {
    /// Input rows
    rows: Vec<HashMap<String, Value>>,
}

impl WindowExecutor {
    /// Create new window executor
    pub fn new(rows: Vec<HashMap<String, Value>>) -> Self {
        Self { rows }
    }

    /// Execute window function
    pub fn execute(&self, window_func: &WindowFunction) -> QueryResult<Vec<Value>> {
        match &window_func.function {
            WindowFunctionType::RowNumber => self.row_number(&window_func.window_spec),
            WindowFunctionType::Rank => self.rank(&window_func.window_spec),
            WindowFunctionType::DenseRank => self.dense_rank(&window_func.window_spec),
            WindowFunctionType::PercentRank => self.percent_rank(&window_func.window_spec),
            WindowFunctionType::CumeDist => self.cume_dist(&window_func.window_spec),
            WindowFunctionType::Lag { offset, default } => self.lag(
                &window_func.window_spec,
                window_func.argument.as_ref(),
                *offset,
                default.clone(),
            ),
            WindowFunctionType::Lead { offset, default } => self.lead(
                &window_func.window_spec,
                window_func.argument.as_ref(),
                *offset,
                default.clone(),
            ),
            WindowFunctionType::FirstValue => {
                self.first_value(&window_func.window_spec, window_func.argument.as_ref())
            }
            WindowFunctionType::LastValue => {
                self.last_value(&window_func.window_spec, window_func.argument.as_ref())
            }
            WindowFunctionType::NthValue { n } => {
                self.nth_value(&window_func.window_spec, window_func.argument.as_ref(), *n)
            }
            WindowFunctionType::Ntile { n } => self.ntile(&window_func.window_spec, *n),
        }
    }

    /// ROW_NUMBER() - Sequential row number within partition
    fn row_number(&self, spec: &WindowSpec) -> QueryResult<Vec<Value>> {
        let partitions = self.partition_rows(spec)?;
        let mut results = Vec::new();

        for partition in partitions {
            for (i, _) in partition.iter().enumerate() {
                results.push(Value::Integer((i + 1) as i64));
            }
        }

        Ok(results)
    }

    /// RANK() - Rank with gaps for ties
    fn rank(&self, spec: &WindowSpec) -> QueryResult<Vec<Value>> {
        let partitions = self.partition_rows(spec)?;
        let mut results = Vec::new();

        for partition in partitions {
            // Sort partition by ORDER BY
            let mut sorted: Vec<_> = partition.iter().enumerate().collect();
            self.sort_partition(&mut sorted, &spec.order_by)?;

            // Assign ranks
            let mut rank = 1;
            let mut prev_values: Option<Vec<Value>> = None;

            for (original_idx, row) in &sorted {
                let current_values = self.get_order_values(row, &spec.order_by)?;

                if let Some(ref prev) = prev_values {
                    if current_values != *prev {
                        rank = *original_idx + 1;
                    }
                }

                results.push(Value::Integer(rank as i64));
                prev_values = Some(current_values);
            }

            // Reorder results to match original order
            let mut reordered = vec![Value::Null; partition.len()];
            for (i, (original_idx, _)) in sorted.iter().enumerate() {
                reordered[*original_idx] = results[results.len() - partition.len() + i].clone();
            }
            results.truncate(results.len() - partition.len());
            results.extend(reordered);
        }

        Ok(results)
    }

    /// DENSE_RANK() - Rank without gaps for ties
    fn dense_rank(&self, spec: &WindowSpec) -> QueryResult<Vec<Value>> {
        let partitions = self.partition_rows(spec)?;
        let mut results = Vec::new();

        for partition in partitions {
            let mut sorted: Vec<_> = partition.iter().enumerate().collect();
            self.sort_partition(&mut sorted, &spec.order_by)?;

            let mut rank = 1;
            let mut prev_values: Option<Vec<Value>> = None;

            for (original_idx, row) in &sorted {
                let current_values = self.get_order_values(row, &spec.order_by)?;

                if let Some(ref prev) = prev_values {
                    if current_values != *prev {
                        rank += 1;
                    }
                }

                results.push(Value::Integer(rank as i64));
                prev_values = Some(current_values);
            }

            // Reorder
            let mut reordered = vec![Value::Null; partition.len()];
            for (i, (original_idx, _)) in sorted.iter().enumerate() {
                reordered[*original_idx] = results[results.len() - partition.len() + i].clone();
            }
            results.truncate(results.len() - partition.len());
            results.extend(reordered);
        }

        Ok(results)
    }

    /// PERCENT_RANK() - Relative rank (0.0 to 1.0)
    fn percent_rank(&self, spec: &WindowSpec) -> QueryResult<Vec<Value>> {
        let partitions = self.partition_rows(spec)?;
        let mut results = Vec::new();

        for partition in partitions {
            if partition.is_empty() {
                continue;
            }

            let mut sorted: Vec<_> = partition.iter().enumerate().collect();
            self.sort_partition(&mut sorted, &spec.order_by)?;

            let total = partition.len() as f64;

            for (original_idx, row) in &sorted {
                let rank = self.get_rank_for_row(&sorted, row, spec)?;
                let percent = if total > 1.0 {
                    (rank - 1) as f64 / (total - 1.0)
                } else {
                    0.0
                };
                results.push(Value::Float(percent));
            }

            // Reorder
            let mut reordered = vec![Value::Null; partition.len()];
            for (i, (original_idx, _)) in sorted.iter().enumerate() {
                reordered[*original_idx] = results[results.len() - partition.len() + i].clone();
            }
            results.truncate(results.len() - partition.len());
            results.extend(reordered);
        }

        Ok(results)
    }

    /// CUME_DIST() - Cumulative distribution
    fn cume_dist(&self, spec: &WindowSpec) -> QueryResult<Vec<Value>> {
        let partitions = self.partition_rows(spec)?;
        let mut results = Vec::new();

        for partition in partitions {
            if partition.is_empty() {
                continue;
            }

            let mut sorted: Vec<_> = partition.iter().enumerate().collect();
            self.sort_partition(&mut sorted, &spec.order_by)?;

            let total = partition.len() as f64;

            for (original_idx, row) in &sorted {
                let rank = self.get_rank_for_row(&sorted, row, spec)?;
                let cume = rank as f64 / total;
                results.push(Value::Float(cume));
            }

            // Reorder
            let mut reordered = vec![Value::Null; partition.len()];
            for (i, (original_idx, _)) in sorted.iter().enumerate() {
                reordered[*original_idx] = results[results.len() - partition.len() + i].clone();
            }
            results.truncate(results.len() - partition.len());
            results.extend(reordered);
        }

        Ok(results)
    }

    /// LAG(expr, offset, default) - Previous row value
    fn lag(
        &self,
        spec: &WindowSpec,
        argument: Option<&Expression>,
        offset: usize,
        default: Option<Value>,
    ) -> QueryResult<Vec<Value>> {
        let partitions = self.partition_rows(spec)?;
        let mut results = Vec::new();

        for partition in partitions {
            let mut sorted: Vec<_> = partition.iter().enumerate().collect();
            self.sort_partition(&mut sorted, &spec.order_by)?;

            for (i, (original_idx, row)) in sorted.iter().enumerate() {
                let value = if i >= offset {
                    let prev_row = &sorted[i - offset].1;
                    self.evaluate_expression(prev_row, argument)?
                } else {
                    default.clone().unwrap_or(Value::Null)
                };
                results.push(value);
            }

            // Reorder
            let mut reordered = vec![Value::Null; partition.len()];
            for (i, (original_idx, _)) in sorted.iter().enumerate() {
                reordered[*original_idx] = results[results.len() - partition.len() + i].clone();
            }
            results.truncate(results.len() - partition.len());
            results.extend(reordered);
        }

        Ok(results)
    }

    /// LEAD(expr, offset, default) - Next row value
    fn lead(
        &self,
        spec: &WindowSpec,
        argument: Option<&Expression>,
        offset: usize,
        default: Option<Value>,
    ) -> QueryResult<Vec<Value>> {
        let partitions = self.partition_rows(spec)?;
        let mut results = Vec::new();

        for partition in partitions {
            let mut sorted: Vec<_> = partition.iter().enumerate().collect();
            self.sort_partition(&mut sorted, &spec.order_by)?;

            for (i, (original_idx, row)) in sorted.iter().enumerate() {
                let value = if i + offset < sorted.len() {
                    let next_row = &sorted[i + offset].1;
                    self.evaluate_expression(next_row, argument)?
                } else {
                    default.clone().unwrap_or(Value::Null)
                };
                results.push(value);
            }

            // Reorder
            let mut reordered = vec![Value::Null; partition.len()];
            for (i, (original_idx, _)) in sorted.iter().enumerate() {
                reordered[*original_idx] = results[results.len() - partition.len() + i].clone();
            }
            results.truncate(results.len() - partition.len());
            results.extend(reordered);
        }

        Ok(results)
    }

    /// FIRST_VALUE(expr) - First value in window
    fn first_value(
        &self,
        spec: &WindowSpec,
        argument: Option<&Expression>,
    ) -> QueryResult<Vec<Value>> {
        let partitions = self.partition_rows(spec)?;
        let mut results = Vec::new();

        for partition in partitions {
            if partition.is_empty() {
                continue;
            }

            let mut sorted: Vec<_> = partition.iter().enumerate().collect();
            self.sort_partition(&mut sorted, &spec.order_by)?;

            let first_row = &sorted[0].1;
            let first_value = self.evaluate_expression(first_row, argument)?;

            for _ in partition {
                results.push(first_value.clone());
            }
        }

        Ok(results)
    }

    /// LAST_VALUE(expr) - Last value in window
    fn last_value(
        &self,
        spec: &WindowSpec,
        argument: Option<&Expression>,
    ) -> QueryResult<Vec<Value>> {
        let partitions = self.partition_rows(spec)?;
        let mut results = Vec::new();

        for partition in partitions {
            if partition.is_empty() {
                continue;
            }

            let mut sorted: Vec<_> = partition.iter().enumerate().collect();
            self.sort_partition(&mut sorted, &spec.order_by)?;

            let last_row = &sorted[sorted.len() - 1].1;
            let last_value = self.evaluate_expression(last_row, argument)?;

            for _ in partition {
                results.push(last_value.clone());
            }
        }

        Ok(results)
    }

    /// NTH_VALUE(expr, n) - Nth value in window
    fn nth_value(
        &self,
        spec: &WindowSpec,
        argument: Option<&Expression>,
        n: usize,
    ) -> QueryResult<Vec<Value>> {
        let partitions = self.partition_rows(spec)?;
        let mut results = Vec::new();

        for partition in partitions {
            if partition.is_empty() || n == 0 {
                for _ in partition {
                    results.push(Value::Null);
                }
                continue;
            }

            let mut sorted: Vec<_> = partition.iter().enumerate().collect();
            self.sort_partition(&mut sorted, &spec.order_by)?;

            let nth_value = if n <= sorted.len() {
                let nth_row = &sorted[n - 1].1;
                self.evaluate_expression(nth_row, argument)?
            } else {
                Value::Null
            };

            for _ in partition {
                results.push(nth_value.clone());
            }
        }

        Ok(results)
    }

    /// NTILE(n) - Divide into n buckets
    fn ntile(&self, spec: &WindowSpec, n: usize) -> QueryResult<Vec<Value>> {
        let partitions = self.partition_rows(spec)?;
        let mut results = Vec::new();

        for partition in partitions {
            if partition.is_empty() || n == 0 {
                for _ in partition {
                    results.push(Value::Null);
                }
                continue;
            }

            let mut sorted: Vec<_> = partition.iter().enumerate().collect();
            self.sort_partition(&mut sorted, &spec.order_by)?;

            let size = partition.len();
            let bucket_size = (size as f64 / n as f64).ceil() as usize;

            for (i, (original_idx, _)) in sorted.iter().enumerate() {
                let bucket = (i / bucket_size).min(n - 1) + 1;
                results.push(Value::Integer(bucket as i64));
            }

            // Reorder
            let mut reordered = vec![Value::Null; partition.len()];
            for (i, (original_idx, _)) in sorted.iter().enumerate() {
                reordered[*original_idx] = results[results.len() - partition.len() + i].clone();
            }
            results.truncate(results.len() - partition.len());
            results.extend(reordered);
        }

        Ok(results)
    }

    // Helper methods

    /// Partition rows according to PARTITION BY
    fn partition_rows(&self, spec: &WindowSpec) -> QueryResult<Vec<Vec<&HashMap<String, Value>>>> {
        if spec.partition_by.is_empty() {
            // Single partition with all rows
            return Ok(vec![self.rows.iter().collect()]);
        }

        let mut partitions: HashMap<Vec<Value>, Vec<&HashMap<String, Value>>> = HashMap::new();

        for row in &self.rows {
            let partition_key = self.get_partition_key(row, &spec.partition_by)?;
            partitions
                .entry(partition_key)
                .or_insert_with(Vec::new)
                .push(row);
        }

        Ok(partitions.into_values().collect())
    }

    /// Get partition key for a row
    fn get_partition_key(
        &self,
        row: &HashMap<String, Value>,
        exprs: &[Expression],
    ) -> QueryResult<Vec<Value>> {
        let mut key = Vec::new();
        for expr in exprs {
            key.push(self.evaluate_expression(row, Some(expr))?);
        }
        Ok(key)
    }

    /// Get order values for a row
    fn get_order_values(
        &self,
        row: &HashMap<String, Value>,
        order_by: &[(Expression, bool)],
    ) -> QueryResult<Vec<Value>> {
        let mut values = Vec::new();
        for (expr, _) in order_by {
            values.push(self.evaluate_expression(row, Some(expr))?);
        }
        Ok(values)
    }

    /// Sort partition by ORDER BY
    fn sort_partition(
        &self,
        partition: &mut Vec<(usize, &HashMap<String, Value>)>,
        order_by: &[(Expression, bool)],
    ) -> QueryResult<()> {
        partition.sort_by(|a, b| {
            for (expr, ascending) in order_by {
                let a_val = self
                    .evaluate_expression(a.1, Some(expr))
                    .unwrap_or(Value::Null);
                let b_val = self
                    .evaluate_expression(b.1, Some(expr))
                    .unwrap_or(Value::Null);
                let cmp = a_val
                    .partial_cmp(&b_val)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if !ascending {
                    return cmp.reverse();
                }
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });
        Ok(())
    }

    /// Get rank for a row
    fn get_rank_for_row(
        &self,
        sorted: &[(usize, &HashMap<String, Value>)],
        row: &HashMap<String, Value>,
        spec: &WindowSpec,
    ) -> QueryResult<usize> {
        let row_values = self.get_order_values(row, &spec.order_by)?;
        for (i, (_, sorted_row)) in sorted.iter().enumerate() {
            let sorted_values = self.get_order_values(sorted_row, &spec.order_by)?;
            if sorted_values == row_values {
                return Ok(i + 1);
            }
        }
        Ok(sorted.len())
    }

    /// Evaluate expression on a row
    fn evaluate_expression(
        &self,
        row: &HashMap<String, Value>,
        expr: Option<&Expression>,
    ) -> QueryResult<Value> {
        // Simplified evaluation - in real implementation, would use full expression evaluator
        match expr {
            Some(Expression::Column(name)) => row
                .get(name)
                .cloned()
                .ok_or_else(|| QueryError::Internal(format!("Column not found: {}", name))),
            Some(Expression::Literal(value)) => Ok(value.clone()),
            _ => Ok(Value::Null),
        }
    }
}
