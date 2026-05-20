//! Storage-backed Query Executor
//!
//! Connects the query planner to actual storage, executing queries
//! against the B-tree engine with support for all SQL operations.

use crate::ast::{Expression, Operator, Query, Value};
use crate::error::{QueryError, QueryResult};
use crate::execution::{ExecutionPlan, QueryContext, ResultSet, Row};
use crate::planner::PlanNode;
use crate::sql::SetOperationType;
use chrono::{Datelike, Timelike};
use std::collections::HashMap;
use std::sync::Arc;

/// Row data - represents a decoded row from storage
#[derive(Debug, Clone)]
pub struct RowData {
    /// Column names
    pub columns: Vec<String>,
    /// Column values
    pub values: Vec<Value>,
}

impl RowData {
    /// Create new row data
    pub fn new(columns: Vec<String>, values: Vec<Value>) -> Self {
        Self { columns, values }
    }

    /// Get value by column name
    pub fn get(&self, column: &str) -> Option<&Value> {
        self.columns
            .iter()
            .position(|c| c == column)
            .and_then(|i| self.values.get(i))
    }

    /// Get value by index
    pub fn get_by_index(&self, index: usize) -> Option<&Value> {
        self.values.get(index)
    }
}

/// Table storage adapter trait - allows different storage backends
pub trait TableStorage: Send + Sync {
    /// Scan all rows in a table
    fn scan(&self, table: &str) -> QueryResult<Vec<RowData>>;

    /// Get table column names
    fn columns(&self, table: &str) -> QueryResult<Vec<String>>;

    /// Insert a row
    fn insert(&self, table: &str, row: &RowData) -> QueryResult<()>;

    /// Update rows matching predicate
    fn update(
        &self,
        table: &str,
        assignments: &HashMap<String, Value>,
        predicate: Option<&Expression>,
    ) -> QueryResult<usize>;

    /// Delete rows matching predicate
    fn delete(&self, table: &str, predicate: Option<&Expression>) -> QueryResult<usize>;

    /// Check if table exists
    fn table_exists(&self, table: &str) -> QueryResult<bool>;

    /// Scan using an index
    fn index_scan(
        &self,
        table: &str,
        index: &str,
        predicate: &Expression,
    ) -> QueryResult<Vec<RowData>> {
        // Default implementation fallback to full scan + filter if not implemented?
        // No, that defeats the purpose. But better than breaking?
        // For now, let's return error to enforce implementation or explicit fallback.
        Err(QueryError::Unsupported(format!(
            "Index scan not supported for table {}",
            table
        )))
    }
}

/// Storage-backed query executor
pub struct StorageExecutor<S: TableStorage> {
    storage: Arc<S>,
}

impl<S: TableStorage> StorageExecutor<S> {
    /// Create new executor with storage backend
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }

    /// Execute a query plan
    pub fn execute(&self, plan: &ExecutionPlan, context: &QueryContext) -> QueryResult<ResultSet> {
        self.execute_node(&plan.root, context)
    }

    /// Execute a plan node recursively
    fn execute_node(&self, node: &PlanNode, context: &QueryContext) -> QueryResult<ResultSet> {
        match node {
            PlanNode::Scan {
                table,
                columns,
                filter,
            } => self.execute_scan(table, columns, filter.as_ref(), context),
            PlanNode::IndexScan {
                table,
                index,
                columns,
                filter,
            } => {
                let rows = self.storage.index_scan(table, index, filter)?;
                let table_columns = self.storage.columns(table)?;

                // Determine which columns to include
                let result_columns: Vec<String> = if columns.is_empty() {
                    table_columns.clone()
                } else {
                    columns.to_vec()
                };

                let mut result = ResultSet::with_columns(result_columns.clone());

                for row_data in rows {
                    // Project columns
                    let values: Vec<Value> = if columns.is_empty() {
                        row_data.values.clone()
                    } else {
                        columns
                            .iter()
                            .map(|col| row_data.get(col).cloned().unwrap_or(Value::Null))
                            .collect()
                    };

                    result.add_row(Row::new(values));
                }

                Ok(result)
            }
            PlanNode::Filter { input, predicate } => {
                let mut result = self.execute_node(input, context)?;
                result = self.apply_filter(result, predicate, context)?;
                Ok(result)
            }
            PlanNode::Project { input, columns } => {
                let mut result = self.execute_node(input, context)?;
                result = self.apply_projection(result, columns)?;
                Ok(result)
            }
            PlanNode::Sort { input, order_by } => {
                let mut result = self.execute_node(input, context)?;
                result = self.apply_sort(result, order_by)?;
                Ok(result)
            }
            PlanNode::Limit {
                input,
                limit,
                offset,
            } => {
                let mut result = self.execute_node(input, context)?;
                result = self.apply_limit(result, *limit, *offset)?;
                Ok(result)
            }
            PlanNode::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                let result = self.execute_node(input, context)?;
                self.apply_aggregation(result, group_by, aggregates)
            }
            PlanNode::Join {
                left,
                right,
                join_type,
                condition,
            } => {
                let left_result = self.execute_node(left, context)?;
                let right_result = self.execute_node(right, context)?;
                self.execute_join(
                    left_result,
                    right_result,
                    join_type,
                    condition.as_ref(),
                    context,
                )
            }
            PlanNode::Window {
                input,
                window_functions,
            } => {
                let mut result = self.execute_node(input, context)?;
                for wf in window_functions {
                    result.columns.push(wf.alias.clone());

                    // Partition rows
                    let partition_cols: Vec<&str> = wf
                        .window
                        .partition_by
                        .iter()
                        .filter_map(|e| {
                            if let Expression::Column(name) = e {
                                Some(name.as_str())
                            } else {
                                None
                            }
                        })
                        .collect();
                    let mut partitions: std::collections::HashMap<String, Vec<usize>> =
                        std::collections::HashMap::new();
                    for (idx, row) in result.rows.iter().enumerate() {
                        let key: String = partition_cols
                            .iter()
                            .map(|col| {
                                let ci = result.columns.iter().position(|c| c == *col);
                                ci.and_then(|i| row.values.get(i))
                                    .map(|v| format!("{:?}", v))
                                    .unwrap_or_default()
                            })
                            .collect::<Vec<_>>()
                            .join("|");
                        partitions.entry(key).or_default().push(idx);
                    }

                    let mut window_values: Vec<Value> = vec![Value::Null; result.rows.len()];
                    for partition_indices in partitions.values() {
                        let sorted_indices = if wf.window.order_by.is_empty() {
                            partition_indices.clone()
                        } else {
                            let mut indices = partition_indices.clone();
                            indices.sort_by(|&a, &b| {
                                for order in &wf.window.order_by {
                                    let va = result
                                        .rows
                                        .get(a)
                                        .map(|r| {
                                            self.eval_order_expr(
                                                &order.expr,
                                                &result.columns,
                                                &r.values,
                                            )
                                        })
                                        .unwrap_or(Value::Null);
                                    let vb = result
                                        .rows
                                        .get(b)
                                        .map(|r| {
                                            self.eval_order_expr(
                                                &order.expr,
                                                &result.columns,
                                                &r.values,
                                            )
                                        })
                                        .unwrap_or(Value::Null);
                                    let cmp = Self::compare_values(&va, &vb);
                                    if cmp != std::cmp::Ordering::Equal {
                                        return if order.descending { cmp.reverse() } else { cmp };
                                    }
                                }
                                std::cmp::Ordering::Equal
                            });
                            indices
                        };

                        match wf.function.to_uppercase().as_str() {
                            "ROW_NUMBER" => {
                                for (i, &idx) in sorted_indices.iter().enumerate() {
                                    window_values[idx] = Value::Int((i + 1) as i64);
                                }
                            }
                            "RANK" => {
                                let mut rank = 1i64;
                                for (i, &idx) in sorted_indices.iter().enumerate() {
                                    if i > 0 {
                                        let prev_idx = sorted_indices[i - 1];
                                        if !Self::window_order_values_equal(
                                            &result,
                                            idx,
                                            prev_idx,
                                            &wf.window.order_by,
                                        ) {
                                            rank = (i + 1) as i64;
                                        }
                                    }
                                    window_values[idx] = Value::Int(rank);
                                }
                            }
                            "DENSE_RANK" => {
                                let mut rank = 1i64;
                                for (i, &idx) in sorted_indices.iter().enumerate() {
                                    if i > 0 {
                                        let prev_idx = sorted_indices[i - 1];
                                        if !Self::window_order_values_equal(
                                            &result,
                                            idx,
                                            prev_idx,
                                            &wf.window.order_by,
                                        ) {
                                            rank += 1;
                                        }
                                    }
                                    window_values[idx] = Value::Int(rank);
                                }
                            }
                            "LAG" => {
                                let offset = wf
                                    .args
                                    .get(1)
                                    .and_then(|e| {
                                        if let Expression::Literal(Value::Int(n)) = e {
                                            Some((*n).max(0) as usize)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(1);
                                let default_val = wf
                                    .args
                                    .get(2)
                                    .and_then(|e| {
                                        if let Expression::Literal(v) = e {
                                            Some(v.clone())
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(Value::Null);
                                let col_idx = wf.args.first().and_then(|e| {
                                    if let Expression::Column(name) = e {
                                        result.columns.iter().position(|c| c == name)
                                    } else {
                                        None
                                    }
                                });
                                for (i, &idx) in sorted_indices.iter().enumerate() {
                                    if i >= offset {
                                        let prev_idx = sorted_indices[i - offset];
                                        window_values[idx] = col_idx
                                            .and_then(|ci| {
                                                result
                                                    .rows
                                                    .get(prev_idx)
                                                    .and_then(|r| r.values.get(ci))
                                            })
                                            .cloned()
                                            .unwrap_or(default_val.clone());
                                    } else {
                                        window_values[idx] = default_val.clone();
                                    }
                                }
                            }
                            "LEAD" => {
                                let offset = wf
                                    .args
                                    .get(1)
                                    .and_then(|e| {
                                        if let Expression::Literal(Value::Int(n)) = e {
                                            Some((*n).max(0) as usize)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(1);
                                let default_val = wf
                                    .args
                                    .get(2)
                                    .and_then(|e| {
                                        if let Expression::Literal(v) = e {
                                            Some(v.clone())
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(Value::Null);
                                let col_idx = wf.args.first().and_then(|e| {
                                    if let Expression::Column(name) = e {
                                        result.columns.iter().position(|c| c == name)
                                    } else {
                                        None
                                    }
                                });
                                for (i, &idx) in sorted_indices.iter().enumerate() {
                                    if i + offset < sorted_indices.len() {
                                        let next_idx = sorted_indices[i + offset];
                                        window_values[idx] = col_idx
                                            .and_then(|ci| {
                                                result
                                                    .rows
                                                    .get(next_idx)
                                                    .and_then(|r| r.values.get(ci))
                                            })
                                            .cloned()
                                            .unwrap_or(default_val.clone());
                                    } else {
                                        window_values[idx] = default_val.clone();
                                    }
                                }
                            }
                            "NTILE" => {
                                let n_buckets = wf
                                    .args
                                    .first()
                                    .and_then(|e| {
                                        if let Expression::Literal(Value::Int(n)) = e {
                                            Some((*n).max(0) as usize)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(1)
                                    .max(1);
                                let total = sorted_indices.len();
                                for (i, &idx) in sorted_indices.iter().enumerate() {
                                    let bucket = if total == 0 {
                                        1
                                    } else {
                                        (i * n_buckets / total) + 1
                                    };
                                    window_values[idx] = Value::Int(bucket as i64);
                                }
                            }
                            "FIRST_VALUE" => {
                                let col_idx = wf.args.first().and_then(|e| {
                                    if let Expression::Column(name) = e {
                                        result.columns.iter().position(|c| c == name)
                                    } else {
                                        None
                                    }
                                });
                                if let Some(ref frame) = wf.window.frame {
                                    for (pos, &idx) in sorted_indices.iter().enumerate() {
                                        let (start, _end) =
                                            frame.frame_range(pos, sorted_indices.len());
                                        let val = col_idx
                                            .and_then(|ci| {
                                                result
                                                    .rows
                                                    .get(sorted_indices[start])
                                                    .and_then(|r| r.values.get(ci))
                                            })
                                            .cloned()
                                            .unwrap_or(Value::Null);
                                        window_values[idx] = val;
                                    }
                                } else {
                                    let first_val = sorted_indices
                                        .first()
                                        .and_then(|&idx| {
                                            col_idx.and_then(|ci| {
                                                result.rows.get(idx).and_then(|r| r.values.get(ci))
                                            })
                                        })
                                        .cloned()
                                        .unwrap_or(Value::Null);
                                    for &idx in &sorted_indices {
                                        window_values[idx] = first_val.clone();
                                    }
                                }
                            }
                            "LAST_VALUE" => {
                                let col_idx = wf.args.first().and_then(|e| {
                                    if let Expression::Column(name) = e {
                                        result.columns.iter().position(|c| c == name)
                                    } else {
                                        None
                                    }
                                });
                                if let Some(ref frame) = wf.window.frame {
                                    for (pos, &idx) in sorted_indices.iter().enumerate() {
                                        let (_start, end) =
                                            frame.frame_range(pos, sorted_indices.len());
                                        let val = col_idx
                                            .and_then(|ci| {
                                                result
                                                    .rows
                                                    .get(sorted_indices[end])
                                                    .and_then(|r| r.values.get(ci))
                                            })
                                            .cloned()
                                            .unwrap_or(Value::Null);
                                        window_values[idx] = val;
                                    }
                                } else {
                                    let last_val = sorted_indices
                                        .last()
                                        .and_then(|&idx| {
                                            col_idx.and_then(|ci| {
                                                result.rows.get(idx).and_then(|r| r.values.get(ci))
                                            })
                                        })
                                        .cloned()
                                        .unwrap_or(Value::Null);
                                    for &idx in &sorted_indices {
                                        window_values[idx] = last_val.clone();
                                    }
                                }
                            }
                            "NTH_VALUE" => {
                                let col_idx = wf.args.first().and_then(|e| {
                                    if let Expression::Column(name) = e {
                                        result.columns.iter().position(|c| c == name)
                                    } else {
                                        None
                                    }
                                });
                                let n = wf
                                    .args
                                    .get(1)
                                    .and_then(|e| {
                                        if let Expression::Literal(Value::Int(n)) = e {
                                            Some((*n).max(0) as usize)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(1);
                                let nth_val = if n >= 1 && n <= sorted_indices.len() {
                                    let target_idx = sorted_indices[n - 1];
                                    col_idx
                                        .and_then(|ci| {
                                            result
                                                .rows
                                                .get(target_idx)
                                                .and_then(|r| r.values.get(ci))
                                        })
                                        .cloned()
                                        .unwrap_or(Value::Null)
                                } else {
                                    Value::Null
                                };
                                for &idx in &sorted_indices {
                                    window_values[idx] = nth_val.clone();
                                }
                            }
                            // ==================== TIME-SERIES WINDOW FUNCTIONS ====================
                            "DELTA" => {
                                // DELTA(value) — current - previous
                                let col_idx = wf.args.first().and_then(|e| {
                                    if let Expression::Column(name) = e {
                                        result.columns.iter().position(|c| c == name)
                                    } else {
                                        None
                                    }
                                });
                                for (i, &idx) in sorted_indices.iter().enumerate() {
                                    if i == 0 {
                                        window_values[idx] = Value::Null;
                                    } else {
                                        let prev_idx = sorted_indices[i - 1];
                                        let cur = col_idx
                                            .and_then(|ci| {
                                                result.rows.get(idx).and_then(|r| r.values.get(ci))
                                            })
                                            .and_then(|v| v.as_float());
                                        let prev = col_idx
                                            .and_then(|ci| {
                                                result
                                                    .rows
                                                    .get(prev_idx)
                                                    .and_then(|r| r.values.get(ci))
                                            })
                                            .and_then(|v| v.as_float());
                                        window_values[idx] = match (cur, prev) {
                                            (Some(c), Some(p)) => Value::Float(c - p),
                                            _ => Value::Null,
                                        };
                                    }
                                }
                            }
                            "RATE" => {
                                // RATE(value, timestamp) — per-second rate of change
                                let val_idx = wf.args.first().and_then(|e| {
                                    if let Expression::Column(name) = e {
                                        result.columns.iter().position(|c| c == name)
                                    } else {
                                        None
                                    }
                                });
                                let ts_idx = wf.args.get(1).and_then(|e| {
                                    if let Expression::Column(name) = e {
                                        result.columns.iter().position(|c| c == name)
                                    } else {
                                        None
                                    }
                                });
                                for (i, &idx) in sorted_indices.iter().enumerate() {
                                    if i == 0 {
                                        window_values[idx] = Value::Null;
                                    } else {
                                        let prev_idx = sorted_indices[i - 1];
                                        let cur_val = val_idx
                                            .and_then(|ci| {
                                                result.rows.get(idx).and_then(|r| r.values.get(ci))
                                            })
                                            .and_then(|v| v.as_float());
                                        let prev_val = val_idx
                                            .and_then(|ci| {
                                                result
                                                    .rows
                                                    .get(prev_idx)
                                                    .and_then(|r| r.values.get(ci))
                                            })
                                            .and_then(|v| v.as_float());
                                        let cur_ts = ts_idx
                                            .and_then(|ci| {
                                                result.rows.get(idx).and_then(|r| r.values.get(ci))
                                            })
                                            .and_then(|v| match v {
                                                Value::Timestamp(t) => Some(*t as f64),
                                                Value::Int(t) => Some(*t as f64),
                                                _ => None,
                                            });
                                        let prev_ts = ts_idx
                                            .and_then(|ci| {
                                                result
                                                    .rows
                                                    .get(prev_idx)
                                                    .and_then(|r| r.values.get(ci))
                                            })
                                            .and_then(|v| match v {
                                                Value::Timestamp(t) => Some(*t as f64),
                                                Value::Int(t) => Some(*t as f64),
                                                _ => None,
                                            });
                                        window_values[idx] =
                                            match (cur_val, prev_val, cur_ts, prev_ts) {
                                                (Some(cv), Some(pv), Some(ct), Some(pt))
                                                    if (ct - pt).abs() > f64::EPSILON =>
                                                {
                                                    Value::Float((cv - pv) / (ct - pt))
                                                }
                                                _ => Value::Null,
                                            };
                                    }
                                }
                            }
                            "LOCF" => {
                                // LOCF(value) — Last Observation Carried Forward
                                let col_idx = wf.args.first().and_then(|e| {
                                    if let Expression::Column(name) = e {
                                        result.columns.iter().position(|c| c == name)
                                    } else {
                                        None
                                    }
                                });
                                let mut last_non_null = Value::Null;
                                for &idx in sorted_indices.iter() {
                                    let val = col_idx
                                        .and_then(|ci| {
                                            result.rows.get(idx).and_then(|r| r.values.get(ci))
                                        })
                                        .cloned()
                                        .unwrap_or(Value::Null);
                                    if !matches!(val, Value::Null) {
                                        last_non_null = val.clone();
                                        window_values[idx] = val;
                                    } else {
                                        window_values[idx] = last_non_null.clone();
                                    }
                                }
                            }
                            "INTERPOLATE" => {
                                // INTERPOLATE(value) — linear interpolation for NULLs
                                let col_idx = wf.args.first().and_then(|e| {
                                    if let Expression::Column(name) = e {
                                        result.columns.iter().position(|c| c == name)
                                    } else {
                                        None
                                    }
                                });
                                // First, collect all values
                                let vals: Vec<Option<f64>> = sorted_indices
                                    .iter()
                                    .map(|&idx| {
                                        col_idx
                                            .and_then(|ci| {
                                                result.rows.get(idx).and_then(|r| r.values.get(ci))
                                            })
                                            .and_then(|v| v.as_float())
                                    })
                                    .collect();
                                // Fill non-nulls directly, interpolate nulls
                                for (i, &idx) in sorted_indices.iter().enumerate() {
                                    if let Some(v) = vals[i] {
                                        window_values[idx] = Value::Float(v);
                                    } else {
                                        // Find previous non-null
                                        let prev =
                                            (0..i).rev().find_map(|j| vals[j].map(|v| (j, v)));
                                        // Find next non-null
                                        let next = ((i + 1)..vals.len())
                                            .find_map(|j| vals[j].map(|v| (j, v)));
                                        window_values[idx] = match (prev, next) {
                                            (Some((pj, pv)), Some((nj, nv))) => {
                                                let frac = (i - pj) as f64 / (nj - pj) as f64;
                                                Value::Float(pv + frac * (nv - pv))
                                            }
                                            (Some((_, pv)), None) => Value::Float(pv), // carry forward
                                            (None, Some((_, nv))) => Value::Float(nv), // carry backward
                                            _ => Value::Null,
                                        };
                                    }
                                }
                            }
                            "SUM" | "AVG" | "COUNT" | "MIN" | "MAX" => {
                                let col_idx = wf.args.first().and_then(|e| {
                                    if let Expression::Column(name) = e {
                                        result.columns.iter().position(|c| c == name)
                                    } else {
                                        None
                                    }
                                });
                                let wf_wildcard = wf
                                    .args
                                    .first()
                                    .map(|a| matches!(a, Expression::Wildcard))
                                    .unwrap_or(true);
                                if let Some(ref frame) = wf.window.frame {
                                    // Frame-aware: compute aggregate per row over its frame window
                                    for (pos, &idx) in sorted_indices.iter().enumerate() {
                                        let (start, end) =
                                            frame.frame_range(pos, sorted_indices.len());
                                        let end = end.min(sorted_indices.len().saturating_sub(1));
                                        let frame_indices = if start <= end && end < sorted_indices.len() {
                                            &sorted_indices[start..=end]
                                        } else {
                                            &[][..]
                                        };
                                        let agg_val = if wf_wildcard
                                            && wf.function.to_uppercase() == "COUNT"
                                        {
                                            Value::Int(frame_indices.len() as i64)
                                        } else {
                                            let vals: Vec<&Value> = frame_indices
                                                .iter()
                                                .filter_map(|&i| {
                                                    col_idx.and_then(|ci| {
                                                        result
                                                            .rows
                                                            .get(i)
                                                            .and_then(|r| r.values.get(ci))
                                                    })
                                                })
                                                .collect();
                                            self.compute_aggregate(&wf.function, &vals, wf_wildcard)
                                                .unwrap_or(Value::Null)
                                        };
                                        window_values[idx] = agg_val;
                                    }
                                } else {
                                    // No frame: aggregate over entire partition
                                    let agg_val =
                                        if wf_wildcard && wf.function.to_uppercase() == "COUNT" {
                                            Value::Int(sorted_indices.len() as i64)
                                        } else {
                                            let vals: Vec<&Value> = sorted_indices
                                                .iter()
                                                .filter_map(|&i| {
                                                    col_idx.and_then(|ci| {
                                                        result
                                                            .rows
                                                            .get(i)
                                                            .and_then(|r| r.values.get(ci))
                                                    })
                                                })
                                                .collect();
                                            self.compute_aggregate(&wf.function, &vals, wf_wildcard)
                                                .unwrap_or(Value::Null)
                                        };
                                    for &idx in &sorted_indices {
                                        window_values[idx] = agg_val.clone();
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    for (row, val) in result.rows.iter_mut().zip(window_values.into_iter()) {
                        row.values.push(val);
                    }
                }
                Ok(result)
            }
            PlanNode::SetOp { left, right, op } => {
                let mut left_result = self.execute_node(left, context)?;
                let right_result = self.execute_node(right, context)?;

                if left_result.columns.len() != right_result.columns.len() {
                    return Err(QueryError::ExecutionError(format!(
                        "each {} query must have the same number of columns: left has {}, right has {}",
                        match op {
                            SetOperationType::Union | SetOperationType::UnionAll => "UNION",
                            SetOperationType::Except | SetOperationType::ExceptAll => "EXCEPT",
                            SetOperationType::Intersect | SetOperationType::IntersectAll =>
                                "INTERSECT",
                        },
                        left_result.columns.len(),
                        right_result.columns.len(),
                    )));
                }

                let row_key = |row: &Row| -> String {
                    row.values
                        .iter()
                        .map(|v| match v {
                            Value::Null => "N".to_string(),
                            Value::Bool(b) => format!("B:{}", b),
                            Value::Int(i) => format!("I:{}", i),
                            Value::Float(f) => format!("F:{}", f.to_bits()),
                            Value::String(s) => format!("S:{}", s),
                            Value::Bytes(b) => format!("Y:{:?}", b),
                            _ => format!("O:{:?}", v),
                        })
                        .collect::<Vec<_>>()
                        .join("\x00")
                };

                match op {
                    SetOperationType::UnionAll => {
                        for row in right_result.rows {
                            left_result.rows.push(row);
                        }
                    }
                    SetOperationType::Union => {
                        for row in right_result.rows {
                            left_result.rows.push(row);
                        }
                        let mut seen = std::collections::HashSet::new();
                        left_result.rows.retain(|row| seen.insert(row_key(row)));
                    }
                    SetOperationType::Except => {
                        let right_keys: std::collections::HashSet<String> =
                            right_result.rows.iter().map(|r| row_key(r)).collect();
                        left_result
                            .rows
                            .retain(|row| !right_keys.contains(&row_key(row)));
                        let mut seen = std::collections::HashSet::new();
                        left_result.rows.retain(|row| seen.insert(row_key(row)));
                    }
                    SetOperationType::ExceptAll => {
                        let mut right_counts: std::collections::HashMap<String, usize> =
                            std::collections::HashMap::new();
                        for row in &right_result.rows {
                            *right_counts.entry(row_key(row)).or_default() += 1;
                        }
                        left_result.rows.retain(|row| {
                            let key = row_key(row);
                            if let Some(count) = right_counts.get_mut(&key) {
                                if *count > 0 {
                                    *count -= 1;
                                    return false;
                                }
                            }
                            true
                        });
                    }
                    SetOperationType::Intersect => {
                        let right_keys: std::collections::HashSet<String> =
                            right_result.rows.iter().map(|r| row_key(r)).collect();
                        left_result
                            .rows
                            .retain(|row| right_keys.contains(&row_key(row)));
                        let mut seen = std::collections::HashSet::new();
                        left_result.rows.retain(|row| seen.insert(row_key(row)));
                    }
                    SetOperationType::IntersectAll => {
                        let mut right_counts: std::collections::HashMap<String, usize> =
                            std::collections::HashMap::new();
                        for row in &right_result.rows {
                            *right_counts.entry(row_key(row)).or_default() += 1;
                        }
                        left_result.rows.retain(|row| {
                            let key = row_key(row);
                            if let Some(count) = right_counts.get_mut(&key) {
                                if *count > 0 {
                                    *count -= 1;
                                    return true;
                                }
                            }
                            false
                        });
                    }
                }
                Ok(left_result)
            }
            PlanNode::Distinct { input } => {
                let mut result = self.execute_node(input, context)?;
                let mut seen = std::collections::HashSet::new();
                result
                    .rows
                    .retain(|row| seen.insert(format!("{:?}", row.values)));
                Ok(result)
            }
            PlanNode::RecursiveCte { base, .. } => {
                // For the planner-based execution path, recursive CTEs are handled
                // at the query level (server/query.rs and storage_executor.rs).
                // When reached through the planner, execute just the base plan.
                self.execute_node(base, context)
            }
            PlanNode::VectorScan { .. } => {
                // Vector scan execution is handled by the HDC/vector index layer.
                // Return empty result when reached through the generic executor.
                Ok(ResultSet::empty())
            }
            PlanNode::WcojJoin { .. } => {
                // WCOJ execution is handled by the wcoj module directly.
                Ok(ResultSet::empty())
            }
            PlanNode::Empty => Ok(ResultSet::empty()),
        }
    }

    /// Execute a table scan
    fn execute_scan(
        &self,
        table: &str,
        columns: &[String],
        filter: Option<&Expression>,
        context: &QueryContext,
    ) -> QueryResult<ResultSet> {
        // Get all rows from storage
        let rows = self.storage.scan(table)?;
        let table_columns = self.storage.columns(table)?;

        // Determine which columns to include
        let result_columns: Vec<String> = if columns.is_empty() {
            table_columns.clone()
        } else {
            columns.to_vec()
        };

        let mut result = ResultSet::with_columns(result_columns.clone());

        for row_data in rows {
            // Apply filter if present
            if let Some(pred) = filter {
                if !self.eval_predicate(pred, &row_data, context)? {
                    continue;
                }
            }

            // Project columns
            let values: Vec<Value> = if columns.is_empty() {
                row_data.values.clone()
            } else {
                columns
                    .iter()
                    .map(|col| row_data.get(col).cloned().unwrap_or(Value::Null))
                    .collect()
            };

            result.add_row(Row::new(values));
        }

        Ok(result)
    }

    /// Apply filter to result set
    fn apply_filter(
        &self,
        mut result: ResultSet,
        predicate: &Expression,
        context: &QueryContext,
    ) -> QueryResult<ResultSet> {
        let filtered_rows: Vec<Row> = result
            .rows
            .into_iter()
            .filter(|row| {
                let row_data = RowData::new(result.columns.clone(), row.values.clone());
                self.eval_predicate(predicate, &row_data, context)
                    .unwrap_or(false)
            })
            .collect();

        result.rows = filtered_rows;
        Ok(result)
    }

    /// Apply projection to result set
    fn apply_projection(
        &self,
        mut result: ResultSet,
        columns: &[String],
    ) -> QueryResult<ResultSet> {
        if columns.is_empty() {
            return Ok(result);
        }

        // Find column indices
        let indices: Vec<Option<usize>> = columns
            .iter()
            .map(|col| result.columns.iter().position(|c| c == col))
            .collect();

        // Project rows
        let projected_rows: Vec<Row> = result
            .rows
            .iter()
            .map(|row| {
                let values: Vec<Value> = indices
                    .iter()
                    .map(|idx| {
                        idx.and_then(|i| row.values.get(i).cloned())
                            .unwrap_or(Value::Null)
                    })
                    .collect();
                Row::new(values)
            })
            .collect();

        result.columns = columns.to_vec();
        result.rows = projected_rows;
        Ok(result)
    }

    /// Apply sort to result set
    fn apply_sort(
        &self,
        mut result: ResultSet,
        order_by: &[crate::ast::OrderBy],
    ) -> QueryResult<ResultSet> {
        if order_by.is_empty() {
            return Ok(result);
        }

        result.rows.sort_by(|a, b| {
            for order in order_by {
                let a_val = self.eval_order_expr(&order.expr, &result.columns, &a.values);
                let b_val = self.eval_order_expr(&order.expr, &result.columns, &b.values);
                let cmp = Self::compare_values(&a_val, &b_val);
                if cmp != std::cmp::Ordering::Equal {
                    return if order.descending { cmp.reverse() } else { cmp };
                }
            }
            std::cmp::Ordering::Equal
        });

        Ok(result)
    }

    /// Evaluate an ORDER BY expression against a row's values
    fn eval_order_expr(&self, expr: &Expression, columns: &[String], values: &[Value]) -> Value {
        match expr {
            Expression::Column(name) => columns
                .iter()
                .position(|c| c == name)
                .and_then(|i| values.get(i).cloned())
                .unwrap_or(Value::Null),
            Expression::Literal(v) => v.clone(),
            Expression::Binary { left, op, right } => {
                let l = self.eval_order_expr(left, columns, values);
                let r = self.eval_order_expr(right, columns, values);
                match (l, op, r) {
                    (Value::Int(a), crate::ast::Operator::Add, Value::Int(b)) => Value::Int(a + b),
                    (Value::Int(a), crate::ast::Operator::Sub, Value::Int(b)) => Value::Int(a - b),
                    (Value::Int(a), crate::ast::Operator::Mul, Value::Int(b)) => Value::Int(a * b),
                    (Value::Int(a), crate::ast::Operator::Div, Value::Int(b)) if b != 0 => {
                        Value::Int(a / b)
                    }
                    (Value::Float(a), crate::ast::Operator::Add, Value::Float(b)) => {
                        Value::Float(a + b)
                    }
                    (Value::Float(a), crate::ast::Operator::Sub, Value::Float(b)) => {
                        Value::Float(a - b)
                    }
                    (Value::Float(a), crate::ast::Operator::Mul, Value::Float(b)) => {
                        Value::Float(a * b)
                    }
                    (Value::Float(a), crate::ast::Operator::Div, Value::Float(b)) if b != 0.0 => {
                        Value::Float(a / b)
                    }
                    (Value::Int(a), crate::ast::Operator::Mul, Value::Float(b))
                    | (Value::Float(b), crate::ast::Operator::Mul, Value::Int(a)) => {
                        Value::Float(a as f64 * b)
                    }
                    (Value::Int(a), crate::ast::Operator::Add, Value::Float(b))
                    | (Value::Float(b), crate::ast::Operator::Add, Value::Int(a)) => {
                        Value::Float(a as f64 + b)
                    }
                    _ => Value::Null,
                }
            }
            Expression::Function { name, args } => {
                let arg_vals: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_order_expr(a, columns, values))
                    .collect();
                match name.to_uppercase().as_str() {
                    "LOWER" => match arg_vals.first() {
                        Some(Value::String(s)) => Value::String(s.to_lowercase()),
                        _ => Value::Null,
                    },
                    "UPPER" => match arg_vals.first() {
                        Some(Value::String(s)) => Value::String(s.to_uppercase()),
                        _ => Value::Null,
                    },
                    "ABS" => match arg_vals.first() {
                        Some(Value::Int(i)) => Value::Int(i.abs()),
                        Some(Value::Float(f)) => Value::Float(f.abs()),
                        _ => Value::Null,
                    },
                    "LENGTH" | "LEN" => match arg_vals.first() {
                        Some(Value::String(s)) => Value::Int(s.len() as i64),
                        _ => Value::Null,
                    },
                    _ => Value::Null,
                }
            }
            _ => Value::Null,
        }
    }

    /// Apply limit and offset
    fn apply_limit(
        &self,
        mut result: ResultSet,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> QueryResult<ResultSet> {
        if let Some(off) = offset {
            if off < result.rows.len() {
                result.rows = result.rows[off..].to_vec();
            } else {
                result.rows.clear();
            }
        }

        if let Some(lim) = limit {
            result.rows.truncate(lim);
        }

        Ok(result)
    }

    /// Apply aggregation
    fn apply_aggregation(
        &self,
        result: ResultSet,
        group_by: &[Expression],
        aggregates: &[(String, String, String)], // (alias, function, column)
    ) -> QueryResult<ResultSet> {
        use std::collections::HashMap;

        if group_by.is_empty() && aggregates.is_empty() {
            return Ok(result);
        }

        // Group rows - use string key since Value doesn't implement Ord
        let mut groups: HashMap<String, (Vec<Value>, Vec<Row>)> = HashMap::new();

        if group_by.is_empty() {
            // Single group for all rows
            groups.insert(String::new(), (vec![], result.rows.clone()));
        } else {
            for row in &result.rows {
                // Build a RowData to evaluate expressions against
                let row_data = RowData::new(result.columns.clone(), row.values.clone());
                let ctx = QueryContext::default();
                let key_values: Vec<Value> = group_by
                    .iter()
                    .map(|expr| {
                        self.eval_expression(expr, &row_data, &ctx)
                            .unwrap_or(Value::Null)
                    })
                    .collect();
                let key_str = format!("{:?}", key_values);
                groups
                    .entry(key_str)
                    .or_insert_with(|| (key_values, Vec::new()))
                    .1
                    .push(row.clone());
            }
        }

        // Build result columns — derive names from expressions
        let mut new_columns: Vec<String> = group_by
            .iter()
            .map(|expr| match expr {
                Expression::Column(name) => name.clone(),
                Expression::Function { name, .. } => name.clone(),
                other => format!("{:?}", other),
            })
            .collect();
        for (alias, _, _) in aggregates {
            new_columns.push(alias.clone());
        }

        let mut new_result = ResultSet::with_columns(new_columns);

        // Compute aggregates for each group
        for (_, (group_key, rows)) in groups {
            let mut values = group_key;

            for (_, func, col) in aggregates {
                let col_idx = result.columns.iter().position(|c| c == col);
                let col_values: Vec<&Value> = rows
                    .iter()
                    .filter_map(|r| col_idx.and_then(|i| r.values.get(i)))
                    .collect();

                let agg_value = self.compute_aggregate(func, &col_values, col == "*")?;
                values.push(agg_value);
            }

            new_result.add_row(Row::new(values));
        }

        Ok(new_result)
    }

    /// Execute a join operation
    fn execute_join(
        &self,
        left: ResultSet,
        right: ResultSet,
        join_type: &crate::ast::JoinType,
        condition: Option<&Expression>,
        context: &QueryContext,
    ) -> QueryResult<ResultSet> {
        use crate::ast::JoinType;

        // Combine column names
        let mut columns = left.columns.clone();
        columns.extend(right.columns.clone());
        let mut result = ResultSet::with_columns(columns.clone());

        match join_type {
            JoinType::Inner => {
                for left_row in &left.rows {
                    for right_row in &right.rows {
                        let combined = self.combine_rows(left_row, right_row);
                        let row_data = RowData::new(columns.clone(), combined.values.clone());

                        let matches = match condition {
                            Some(cond) => self.eval_predicate(cond, &row_data, context)?,
                            None => true,
                        };

                        if matches {
                            result.add_row(combined);
                        }
                    }
                }
            }
            JoinType::Left => {
                for left_row in &left.rows {
                    let mut matched = false;

                    for right_row in &right.rows {
                        let combined = self.combine_rows(left_row, right_row);
                        let row_data = RowData::new(columns.clone(), combined.values.clone());

                        let matches = match condition {
                            Some(cond) => self.eval_predicate(cond, &row_data, context)?,
                            None => true,
                        };

                        if matches {
                            result.add_row(combined);
                            matched = true;
                        }
                    }

                    if !matched {
                        // Add left row with NULLs for right columns
                        let null_right: Vec<Value> =
                            (0..right.columns.len()).map(|_| Value::Null).collect();
                        let mut values = left_row.values.clone();
                        values.extend(null_right);
                        result.add_row(Row::new(values));
                    }
                }
            }
            JoinType::Right => {
                for right_row in &right.rows {
                    let mut matched = false;

                    for left_row in &left.rows {
                        let combined = self.combine_rows(left_row, right_row);
                        let row_data = RowData::new(columns.clone(), combined.values.clone());

                        let matches = match condition {
                            Some(cond) => self.eval_predicate(cond, &row_data, context)?,
                            None => true,
                        };

                        if matches {
                            result.add_row(combined);
                            matched = true;
                        }
                    }

                    if !matched {
                        let null_left: Vec<Value> =
                            (0..left.columns.len()).map(|_| Value::Null).collect();
                        let mut values = null_left;
                        values.extend(right_row.values.clone());
                        result.add_row(Row::new(values));
                    }
                }
            }
            JoinType::Full => {
                // Track which right rows matched
                let mut right_matched = vec![false; right.rows.len()];

                for left_row in &left.rows {
                    let mut matched = false;

                    for (ri, right_row) in right.rows.iter().enumerate() {
                        let combined = self.combine_rows(left_row, right_row);
                        let row_data = RowData::new(columns.clone(), combined.values.clone());

                        let matches = match condition {
                            Some(cond) => self.eval_predicate(cond, &row_data, context)?,
                            None => true,
                        };

                        if matches {
                            result.add_row(combined);
                            matched = true;
                            right_matched[ri] = true;
                        }
                    }

                    if !matched {
                        let null_right: Vec<Value> =
                            (0..right.columns.len()).map(|_| Value::Null).collect();
                        let mut values = left_row.values.clone();
                        values.extend(null_right);
                        result.add_row(Row::new(values));
                    }
                }

                // Add unmatched right rows
                for (ri, right_row) in right.rows.iter().enumerate() {
                    if !right_matched[ri] {
                        let null_left: Vec<Value> =
                            (0..left.columns.len()).map(|_| Value::Null).collect();
                        let mut values = null_left;
                        values.extend(right_row.values.clone());
                        result.add_row(Row::new(values));
                    }
                }
            }
            JoinType::Cross => {
                // Cartesian product
                for left_row in &left.rows {
                    for right_row in &right.rows {
                        result.add_row(self.combine_rows(left_row, right_row));
                    }
                }
            }
        }

        Ok(result)
    }

    // Helper functions

    fn combine_rows(&self, left: &Row, right: &Row) -> Row {
        let mut values = left.values.clone();
        values.extend(right.values.clone());
        Row::new(values)
    }

    fn eval_predicate(
        &self,
        expr: &Expression,
        row: &RowData,
        context: &QueryContext,
    ) -> QueryResult<bool> {
        let value = self.eval_expression(expr, row, context)?;
        match value {
            Value::Bool(b) => Ok(b),
            Value::Null => Ok(false),
            _ => Ok(true), // Non-null values are truthy
        }
    }

    fn execute_subquery_expr(
        &self,
        query: &Query,
        outer_row: &RowData,
        context: &QueryContext,
    ) -> QueryResult<Vec<RowData>> {
        if context.subquery_depth >= crate::execution::MAX_SUBQUERY_DEPTH {
            return Err(QueryError::ExecutionError(format!(
                "Subquery nesting depth exceeds maximum ({})",
                crate::execution::MAX_SUBQUERY_DEPTH
            )));
        }
        let deeper = QueryContext {
            subquery_depth: context.subquery_depth + 1,
            ..context.clone()
        };
        let context = &deeper;
        let table_name = match &query.source {
            Some(name) => name.as_str(),
            None => return Ok(Vec::new()),
        };

        let rows = self.storage.scan(table_name)?;

        let mut result = Vec::new();

        for inner_row in &rows {
            // Build combined row for correlated subqueries
            let mut combined_cols = outer_row.columns.clone();
            combined_cols.extend(inner_row.columns.iter().cloned());
            let mut combined_vals = outer_row.values.clone();
            combined_vals.extend(inner_row.values.iter().cloned());
            let combined = RowData::new(combined_cols, combined_vals);

            // Apply WHERE filter
            let matches = if let Some(ref filter) = query.filter {
                self.eval_predicate(filter, &combined, context)?
            } else {
                true
            };

            if matches {
                // Project columns
                if query.columns.contains(&"*".to_string()) || query.columns.is_empty() {
                    result.push(inner_row.clone());
                } else {
                    let proj_cols: Vec<String> = query.columns.clone();
                    let proj_vals: Vec<Value> = query
                        .columns
                        .iter()
                        .map(|col| {
                            if let Some(expr) = query.derived_columns.get(col) {
                                self.eval_expression(expr, &combined, context)
                                    .unwrap_or(Value::Null)
                            } else {
                                inner_row.get(col).cloned().unwrap_or(Value::Null)
                            }
                        })
                        .collect();
                    result.push(RowData::new(proj_cols, proj_vals));
                }
            }
        }

        if let Some(limit) = query.limit {
            result.truncate(limit);
        }

        Ok(result)
    }

    fn eval_expression(
        &self,
        expr: &Expression,
        row: &RowData,
        context: &QueryContext,
    ) -> QueryResult<Value> {
        match expr {
            Expression::Literal(v) => Ok(v.clone()),
            Expression::Column(name) => Ok(row.get(name).cloned().unwrap_or(Value::Null)),
            Expression::QualifiedColumn { table: _, column } => {
                // For now, ignore table qualifier
                Ok(row.get(column).cloned().unwrap_or(Value::Null))
            }
            Expression::Binary { left, op, right } => {
                let lval = self.eval_expression(left, row, context)?;
                let rval = self.eval_expression(right, row, context)?;
                self.eval_binary_op(&lval, op, &rval)
            }
            Expression::Unary { op, expr } => {
                let val = self.eval_expression(expr, row, context)?;
                self.eval_unary_op(op, &val)
            }
            Expression::IsNull { expr, negated } => {
                let val = self.eval_expression(expr, row, context)?;
                let is_null = matches!(val, Value::Null);
                Ok(Value::Bool(if *negated { !is_null } else { is_null }))
            }
            Expression::In {
                expr,
                list,
                negated,
            } => {
                let val = self.eval_expression(expr, row, context)?;
                let mut found = false;
                for item in list {
                    if let Expression::Subquery(subquery) = item {
                        let results = self.execute_subquery_expr(subquery, row, context)?;
                        if results.iter().any(|r| {
                            r.values
                                .first()
                                .map_or(false, |v| Self::values_equal(&val, v))
                        }) {
                            found = true;
                            break;
                        }
                    } else {
                        let item_val = self.eval_expression(item, row, context)?;
                        if Self::values_equal(&val, &item_val) {
                            found = true;
                            break;
                        }
                    }
                }
                Ok(Value::Bool(if *negated { !found } else { found }))
            }
            Expression::Between {
                expr,
                low,
                high,
                negated,
            } => {
                let val = self.eval_expression(expr, row, context)?;
                let low_val = self.eval_expression(low, row, context)?;
                let high_val = self.eval_expression(high, row, context)?;

                let in_range = Self::compare_values(&val, &low_val) != std::cmp::Ordering::Less
                    && Self::compare_values(&val, &high_val) != std::cmp::Ordering::Greater;

                Ok(Value::Bool(if *negated { !in_range } else { in_range }))
            }
            Expression::Like {
                expr,
                pattern,
                negated,
                case_insensitive,
            } => {
                let val = self.eval_expression(expr, row, context)?;
                if let Value::String(s) = val {
                    let (s, pat) = if *case_insensitive {
                        (s.to_lowercase(), pattern.to_lowercase())
                    } else {
                        (s, pattern.clone())
                    };
                    let regex_pattern = pat.replace('%', ".*").replace('_', ".");
                    let matches = regex::Regex::new(&format!("^{}$", regex_pattern))
                        .map(|re| re.is_match(&s))
                        .unwrap_or(false);
                    Ok(Value::Bool(if *negated { !matches } else { matches }))
                } else {
                    Ok(Value::Bool(false))
                }
            }
            Expression::Function { name, args } => {
                let arg_values: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expression(a, row, context))
                    .collect::<QueryResult<Vec<_>>>()?;
                self.eval_function(name, &arg_values)
            }
            Expression::Parameter(idx) => Ok(context
                .positional_params
                .get(*idx)
                .cloned()
                .unwrap_or(Value::Null)),
            Expression::NamedParameter(name) => {
                Ok(context.parameters.get(name).cloned().unwrap_or(Value::Null))
            }
            Expression::Cast { expr, target_type } => {
                let val = self.eval_expression(expr, row, context)?;
                Self::cast_value(&val, target_type)
            }
            Expression::Wildcard => Ok(Value::Null),
            Expression::Subquery(subquery) => {
                let results = self.execute_subquery_expr(subquery, row, context)?;
                if let Some(first_row) = results.first() {
                    Ok(first_row.values.first().cloned().unwrap_or(Value::Null))
                } else {
                    Ok(Value::Null)
                }
            }
            Expression::Exists(subquery) => {
                let results = self.execute_subquery_expr(subquery, row, context)?;
                Ok(Value::Bool(!results.is_empty()))
            }
            Expression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(op) = operand {
                    let op_val = self.eval_expression(op, row, context)?;
                    for (when_expr, then_expr) in when_clauses {
                        let when_val = self.eval_expression(when_expr, row, context)?;
                        if Self::values_equal(&op_val, &when_val) {
                            return self.eval_expression(then_expr, row, context);
                        }
                    }
                } else {
                    for (when_expr, then_expr) in when_clauses {
                        let when_val = self.eval_expression(when_expr, row, context)?;
                        if matches!(when_val, Value::Bool(true)) {
                            return self.eval_expression(then_expr, row, context);
                        }
                    }
                }
                if let Some(else_expr) = else_clause {
                    self.eval_expression(else_expr, row, context)
                } else {
                    Ok(Value::Null)
                }
            }
            _ => Ok(Value::Null),
        }
    }

    fn cast_value(val: &Value, target_type: &str) -> QueryResult<Value> {
        match target_type {
            "INT" | "INTEGER" | "BIGINT" => match val {
                Value::Int(_) => Ok(val.clone()),
                Value::Float(f) => Ok(Value::Int(*f as i64)),
                Value::String(s) => s
                    .parse::<i64>()
                    .map(Value::Int)
                    .or_else(|_| s.parse::<f64>().map(|f| Value::Int(f as i64)))
                    .map_err(|_| QueryError::TypeError(format!("Cannot cast '{}' to INT", s))),
                Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
                Value::Null => Ok(Value::Null),
                _ => Ok(Value::Null),
            },
            "FLOAT" | "DOUBLE" | "REAL" | "NUMERIC" | "DECIMAL" => match val {
                Value::Float(_) => Ok(val.clone()),
                Value::Int(i) => Ok(Value::Float(*i as f64)),
                Value::String(s) => s
                    .parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| QueryError::TypeError(format!("Cannot cast '{}' to FLOAT", s))),
                Value::Bool(b) => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
                Value::Null => Ok(Value::Null),
                _ => Ok(Value::Null),
            },
            "TEXT" | "VARCHAR" | "STRING" | "CHAR" => match val {
                Value::String(_) => Ok(val.clone()),
                Value::Int(i) => Ok(Value::String(i.to_string())),
                Value::Float(f) => Ok(Value::String(f.to_string())),
                Value::Bool(b) => Ok(Value::String(b.to_string())),
                Value::Null => Ok(Value::Null),
                _ => Ok(Value::String(format!("{:?}", val))),
            },
            "BOOL" | "BOOLEAN" => match val {
                Value::Bool(_) => Ok(val.clone()),
                Value::Int(i) => Ok(Value::Bool(*i != 0)),
                Value::Float(f) => Ok(Value::Bool(*f != 0.0)),
                Value::String(s) => match s.to_uppercase().as_str() {
                    "TRUE" | "1" | "YES" | "T" => Ok(Value::Bool(true)),
                    "FALSE" | "0" | "NO" | "F" => Ok(Value::Bool(false)),
                    _ => Err(QueryError::TypeError(format!(
                        "Cannot cast '{}' to BOOL",
                        s
                    ))),
                },
                Value::Null => Ok(Value::Null),
                _ => Ok(Value::Null),
            },
            _ => Err(QueryError::TypeError(format!(
                "Unknown target type: {}",
                target_type
            ))),
        }
    }

    fn eval_binary_op(&self, left: &Value, op: &Operator, right: &Value) -> QueryResult<Value> {
        crate::functions::eval_binary_op(left, op, right)
    }

    fn eval_unary_op(&self, op: &crate::ast::UnaryOperator, val: &Value) -> QueryResult<Value> {
        crate::functions::eval_unary_op(op, val)
    }

    fn numeric_op<F, G>(
        &self,
        left: &Value,
        right: &Value,
        int_op: F,
        float_op: G,
    ) -> QueryResult<Value>
    where
        F: Fn(i64, i64) -> i64,
        G: Fn(f64, f64) -> f64,
    {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(int_op(*a, *b))),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(float_op(*a, *b))),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(float_op(*a as f64, *b))),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(float_op(*a, *b as f64))),
            _ => Ok(Value::Null),
        }
    }

    fn eval_function(&self, name: &str, args: &[Value]) -> QueryResult<Value> {
        let storage = &self.storage;
        let scanner = |table: &str| -> QueryResult<(Vec<String>, Vec<Vec<serde_json::Value>>)> {
            let rows = storage.scan(table)?;
            let columns = if rows.is_empty() {
                storage.columns(table)?
            } else {
                rows[0].columns.clone()
            };
            let json_rows = rows
                .iter()
                .map(|r| {
                    r.values
                        .iter()
                        .map(crate::functions::ast_value_to_json)
                        .collect()
                })
                .collect();
            Ok((columns, json_rows))
        };
        crate::functions::eval_scalar_function(name, args, Some(&scanner))
    }

    /// Convert a serde_json::Value to crate::ast::Value
    fn serde_json_to_value(v: serde_json::Value) -> Value {
        match v {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else if let Some(f) = n.as_f64() {
                    Value::Float(f)
                } else {
                    Value::Null
                }
            }
            serde_json::Value::String(s) => Value::String(s),
            serde_json::Value::Array(arr) => {
                Value::Array(arr.into_iter().map(Self::serde_json_to_value).collect())
            }
            serde_json::Value::Object(map) => Value::Object(
                map.into_iter()
                    .map(|(k, v)| (k, Self::serde_json_to_value(v)))
                    .collect(),
            ),
        }
    }

    /// Convert a crate::ast::Value to serde_json::Value (for graph function interop)
    fn ast_value_to_json(v: &Value) -> serde_json::Value {
        match v {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::json!(b),
            Value::Int(i) => serde_json::json!(i),
            Value::Float(f) => serde_json::json!(f),
            Value::String(s) => serde_json::json!(s),
            Value::Timestamp(t) => serde_json::json!(t),
            Value::Uuid(u) => serde_json::json!(u),
            Value::Bytes(b) => serde_json::json!(format!("{:?}", b)),
            Value::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(Self::ast_value_to_json).collect())
            }
            Value::Object(obj) => {
                let map: serde_json::Map<String, serde_json::Value> = obj
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::ast_value_to_json(v)))
                    .collect();
                serde_json::Value::Object(map)
            }
            Value::Vector(v) => serde_json::json!(v),
        }
    }

    fn compute_aggregate(
        &self,
        func: &str,
        values: &[&Value],
        is_wildcard: bool,
    ) -> QueryResult<Value> {
        let func_upper = func.to_uppercase();
        match func_upper.as_str() {
            "COUNT" => {
                if is_wildcard {
                    Ok(Value::Int(values.len() as i64))
                } else {
                    Ok(Value::Int(
                        values.iter().filter(|v| !matches!(v, Value::Null)).count() as i64,
                    ))
                }
            }
            "SUM" => {
                let nums: Vec<f64> = values.iter().filter_map(|v| v.as_float()).collect();
                if nums.is_empty() {
                    Ok(Value::Null)
                } else {
                    Ok(Value::Float(nums.iter().sum()))
                }
            }
            "AVG" => {
                let vals: Vec<f64> = values.iter().filter_map(|v| v.as_float()).collect();
                if vals.is_empty() {
                    Ok(Value::Null)
                } else {
                    Ok(Value::Float(vals.iter().sum::<f64>() / vals.len() as f64))
                }
            }
            "MIN" => values
                .iter()
                .min_by(|a, b| Self::compare_values(a, b))
                .map(|v| (*v).clone())
                .ok_or_else(|| QueryError::ExecutionError("MIN on empty set".to_string()))
                .or(Ok(Value::Null)),
            "MAX" => values
                .iter()
                .max_by(|a, b| Self::compare_values(a, b))
                .map(|v| (*v).clone())
                .ok_or_else(|| QueryError::ExecutionError("MAX on empty set".to_string()))
                .or(Ok(Value::Null)),
            "FIRST" => Ok(values.first().map(|v| (*v).clone()).unwrap_or(Value::Null)),
            "LAST" => Ok(values.last().map(|v| (*v).clone()).unwrap_or(Value::Null)),
            "COUNT_DISTINCT" => {
                let mut unique: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                for v in values {
                    if !matches!(v, Value::Null) {
                        unique.insert(format!("{:?}", v));
                    }
                }
                Ok(Value::Int(unique.len() as i64))
            }
            "STRING_AGG" | "GROUP_CONCAT" => {
                let parts: Vec<String> = values
                    .iter()
                    .filter_map(|v| match v {
                        Value::String(s) => Some(s.clone()),
                        Value::Null => None,
                        Value::Int(n) => Some(n.to_string()),
                        Value::Float(f) => Some(f.to_string()),
                        Value::Bool(b) => Some(b.to_string()),
                        other => Some(format!("{:?}", other)),
                    })
                    .collect();
                if parts.is_empty() {
                    Ok(Value::Null)
                } else {
                    Ok(Value::String(parts.join(",")))
                }
            }
            "JSON_AGG" => {
                let collected: Vec<Value> = values
                    .iter()
                    .filter(|v| !matches!(v, Value::Null))
                    .map(|v| (*v).clone())
                    .collect();
                Ok(Value::Array(collected))
            }
            "JSON_OBJECT_AGG" => {
                // JSON_OBJECT_AGG collects pairs of (key, value) into a JSON object
                // values come interleaved: [key1, val1, key2, val2, ...]
                let mut map = serde_json::Map::new();
                let mut i = 0;
                while i + 1 < values.len() {
                    let key = match &values[i] {
                        Value::String(s) => s.clone(),
                        Value::Int(n) => n.to_string(),
                        _ => format!("{:?}", values[i]),
                    };
                    map.insert(key, crate::ast::value_to_serde_json(&values[i + 1]));
                    i += 2;
                }
                Ok(Value::String(serde_json::Value::Object(map).to_string()))
            }
            "STDDEV" | "STDDEV_POP" => {
                let nums: Vec<f64> = values.iter().filter_map(|v| v.as_float()).collect();
                if nums.is_empty() {
                    Ok(Value::Null)
                } else {
                    let mean = nums.iter().sum::<f64>() / nums.len() as f64;
                    let variance =
                        nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nums.len() as f64;
                    Ok(Value::Float(variance.sqrt()))
                }
            }
            "STDDEV_SAMP" => {
                let nums: Vec<f64> = values.iter().filter_map(|v| v.as_float()).collect();
                if nums.len() < 2 {
                    Ok(Value::Null)
                } else {
                    let mean = nums.iter().sum::<f64>() / nums.len() as f64;
                    let variance = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / (nums.len() - 1) as f64;
                    Ok(Value::Float(variance.sqrt()))
                }
            }
            "VARIANCE" | "VAR_POP" => {
                let nums: Vec<f64> = values.iter().filter_map(|v| v.as_float()).collect();
                if nums.is_empty() {
                    Ok(Value::Null)
                } else {
                    let mean = nums.iter().sum::<f64>() / nums.len() as f64;
                    let variance =
                        nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nums.len() as f64;
                    Ok(Value::Float(variance))
                }
            }
            "VAR_SAMP" => {
                let nums: Vec<f64> = values.iter().filter_map(|v| v.as_float()).collect();
                if nums.len() < 2 {
                    Ok(Value::Null)
                } else {
                    let mean = nums.iter().sum::<f64>() / nums.len() as f64;
                    let variance = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / (nums.len() - 1) as f64;
                    Ok(Value::Float(variance))
                }
            }
            "MEDIAN" => {
                let mut nums: Vec<f64> = values.iter().filter_map(|v| v.as_float()).collect();
                if nums.is_empty() {
                    Ok(Value::Null)
                } else {
                    nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let mid = nums.len() / 2;
                    if nums.len() % 2 == 0 {
                        Ok(Value::Float((nums[mid - 1] + nums[mid]) / 2.0))
                    } else {
                        Ok(Value::Float(nums[mid]))
                    }
                }
            }
            "APPROX_COUNT_DISTINCT" => {
                // HyperLogLog approximate count distinct
                let precision = 14u8;
                let m = 1usize << precision;
                let mut registers = vec![0u8; m];
                for val in values {
                    if matches!(val, Value::Null) {
                        continue;
                    }
                    let hash = {
                        let s = format!("{:?}", val);
                        let mut h: u64 = 0xcbf29ce484222325;
                        for b in s.bytes() {
                            h ^= b as u64;
                            h = h.wrapping_mul(0x100000001b3);
                        }
                        h
                    };
                    let idx = (hash as usize) & (m - 1);
                    let w = hash >> precision;
                    let rho = if w == 0 {
                        (64 - precision) as u8 + 1
                    } else {
                        (w.leading_zeros() as u8) + 1
                    };
                    if rho > registers[idx] {
                        registers[idx] = rho;
                    }
                }
                let alpha_m = 0.7213 / (1.0 + 1.079 / m as f64);
                let raw: f64 = alpha_m * (m as f64) * (m as f64)
                    / registers
                        .iter()
                        .map(|&r| 2.0f64.powi(-(r as i32)))
                        .sum::<f64>();
                let estimate = if raw <= 2.5 * m as f64 {
                    let zeros = registers.iter().filter(|&&r| r == 0).count();
                    if zeros > 0 {
                        (m as f64) * ((m as f64) / zeros as f64).ln()
                    } else {
                        raw
                    }
                } else {
                    raw
                };
                Ok(Value::Int(estimate.round() as i64))
            }
            "APPROX_PERCENTILE" | "PERCENTILE_APPROX" => {
                // Approximate percentile — defaults to 0.5 (median) when percentile not specified
                let percentile = 0.5f64;
                let mut nums: Vec<f64> = values
                    .iter()
                    .filter_map(|v| match v {
                        Value::Float(f) => Some(*f),
                        Value::Int(i) => Some(*i as f64),
                        _ => None,
                    })
                    .collect();
                if nums.is_empty() {
                    return Ok(Value::Null);
                }
                nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = (percentile * (nums.len() - 1) as f64).round() as usize;
                let idx = idx.min(nums.len() - 1);
                Ok(Value::Float(nums[idx]))
            }
            _ => Err(QueryError::UnknownFunction(func.to_string())),
        }
    }

    fn values_equal(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
            (Value::Int(a), Value::Float(b)) => (*a as f64 - b).abs() < f64::EPSILON,
            (Value::Float(a), Value::Int(b)) => (a - *b as f64).abs() < f64::EPSILON,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Bytes(a), Value::Bytes(b)) => a == b,
            _ => false,
        }
    }

    fn window_order_values_equal(
        result: &ResultSet,
        idx_a: usize,
        idx_b: usize,
        order_by: &[crate::ast::OrderBy],
    ) -> bool {
        for ob in order_by {
            if let Expression::Column(name) = &ob.expr {
                if let Some(ci) = result.columns.iter().position(|c| c == name) {
                    let va = result.rows.get(idx_a).and_then(|r| r.values.get(ci));
                    let vb = result.rows.get(idx_b).and_then(|r| r.values.get(ci));
                    match (va, vb) {
                        (Some(a), Some(b)) => {
                            if !Self::values_equal(a, b) {
                                return false;
                            }
                        }
                        (None, None) => {}
                        _ => return false,
                    }
                }
            }
        }
        true
    }

    fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (a, b) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Null, _) => Ordering::Less,
            (_, Value::Null) => Ordering::Greater,
            (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (Value::Int(a), Value::Float(b)) => {
                (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (Value::Float(a), Value::Int(b)) => {
                a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal)
            }
            (Value::String(a), Value::String(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::RwLock;

    /// In-memory table storage for testing
    struct MemoryTableStorage {
        tables: RwLock<HashMap<String, (Vec<String>, Vec<RowData>)>>,
    }

    impl MemoryTableStorage {
        fn new() -> Self {
            Self {
                tables: RwLock::new(HashMap::new()),
            }
        }

        fn create_table(&self, name: &str, columns: Vec<String>) {
            let mut tables = self.tables.write().unwrap();
            tables.insert(name.to_string(), (columns, Vec::new()));
        }

        fn insert_row(&self, table: &str, values: Vec<Value>) {
            let mut tables = self.tables.write().unwrap();
            if let Some((cols, rows)) = tables.get_mut(table) {
                rows.push(RowData::new(cols.clone(), values));
            }
        }
    }

    impl TableStorage for MemoryTableStorage {
        fn scan(&self, table: &str) -> QueryResult<Vec<RowData>> {
            let tables = self.tables.read().unwrap();
            tables
                .get(table)
                .map(|(_, rows)| rows.clone())
                .ok_or_else(|| QueryError::UnknownTable(table.to_string()))
        }

        fn columns(&self, table: &str) -> QueryResult<Vec<String>> {
            let tables = self.tables.read().unwrap();
            tables
                .get(table)
                .map(|(cols, _)| cols.clone())
                .ok_or_else(|| QueryError::UnknownTable(table.to_string()))
        }

        fn insert(&self, table: &str, row: &RowData) -> QueryResult<()> {
            let mut tables = self.tables.write().unwrap();
            if let Some((_, rows)) = tables.get_mut(table) {
                rows.push(row.clone());
                Ok(())
            } else {
                Err(QueryError::UnknownTable(table.to_string()))
            }
        }

        fn update(
            &self,
            table: &str,
            assignments: &HashMap<String, Value>,
            _predicate: Option<&Expression>,
        ) -> QueryResult<usize> {
            let mut tables = self.tables.write().unwrap();
            let (cols, rows) = tables
                .get_mut(table)
                .ok_or_else(|| QueryError::UnknownTable(table.to_string()))?;
            // Apply assignments to all rows (predicate evaluation not implemented for test storage)
            let mut count = 0;
            for row in rows.iter_mut() {
                for (col_name, new_val) in assignments {
                    if let Some(col_idx) = cols.iter().position(|c| c == col_name) {
                        row.values[col_idx] = new_val.clone();
                    }
                }
                count += 1;
            }
            Ok(count)
        }

        fn delete(&self, table: &str, _predicate: Option<&Expression>) -> QueryResult<usize> {
            let mut tables = self.tables.write().unwrap();
            let (_, rows) = tables
                .get_mut(table)
                .ok_or_else(|| QueryError::UnknownTable(table.to_string()))?;
            // Delete all rows (predicate evaluation not implemented for test storage)
            let count = rows.len();
            rows.clear();
            Ok(count)
        }

        fn table_exists(&self, table: &str) -> QueryResult<bool> {
            let tables = self.tables.read().unwrap();
            Ok(tables.contains_key(table))
        }
    }

    fn setup_test_data() -> Arc<MemoryTableStorage> {
        let storage = Arc::new(MemoryTableStorage::new());

        storage.create_table(
            "users",
            vec!["id".to_string(), "name".to_string(), "age".to_string()],
        );
        storage.insert_row(
            "users",
            vec![
                Value::Int(1),
                Value::String("Alice".to_string()),
                Value::Int(30),
            ],
        );
        storage.insert_row(
            "users",
            vec![
                Value::Int(2),
                Value::String("Bob".to_string()),
                Value::Int(25),
            ],
        );
        storage.insert_row(
            "users",
            vec![
                Value::Int(3),
                Value::String("Charlie".to_string()),
                Value::Int(35),
            ],
        );

        storage.create_table(
            "orders",
            vec![
                "id".to_string(),
                "user_id".to_string(),
                "amount".to_string(),
            ],
        );
        storage.insert_row(
            "orders",
            vec![Value::Int(1), Value::Int(1), Value::Float(100.0)],
        );
        storage.insert_row(
            "orders",
            vec![Value::Int(2), Value::Int(1), Value::Float(200.0)],
        );
        storage.insert_row(
            "orders",
            vec![Value::Int(3), Value::Int(2), Value::Float(150.0)],
        );

        storage
    }

    #[test]
    fn test_simple_scan() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        let plan = ExecutionPlan::new(PlanNode::Scan {
            table: "users".to_string(),
            columns: vec![],
            filter: None,
        });

        let result = executor.execute(&plan, &context).unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.columns, vec!["id", "name", "age"]);
    }

    #[test]
    fn test_scan_with_filter() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        let filter = Expression::Binary {
            left: Box::new(Expression::Column("age".to_string())),
            op: Operator::Gt,
            right: Box::new(Expression::Literal(Value::Int(25))),
        };

        let plan = ExecutionPlan::new(PlanNode::Scan {
            table: "users".to_string(),
            columns: vec![],
            filter: Some(filter),
        });

        let result = executor.execute(&plan, &context).unwrap();
        assert_eq!(result.rows.len(), 2); // Alice (30) and Charlie (35)
    }

    #[test]
    fn test_projection() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        let plan = ExecutionPlan::new(
            PlanNode::Scan {
                table: "users".to_string(),
                columns: vec![],
                filter: None,
            }
            .project(vec!["name".to_string(), "age".to_string()]),
        );

        let result = executor.execute(&plan, &context).unwrap();
        assert_eq!(result.columns, vec!["name", "age"]);
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn test_sort() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        let plan = ExecutionPlan::new(
            PlanNode::Scan {
                table: "users".to_string(),
                columns: vec![],
                filter: None,
            }
            .sort(vec![crate::ast::OrderBy {
                expr: Expression::Column("age".to_string()),
                descending: true,
                nulls_first: None,
            }]),
        );

        let result = executor.execute(&plan, &context).unwrap();
        assert_eq!(result.rows[0].values[2], Value::Int(35)); // Charlie first
        assert_eq!(result.rows[2].values[2], Value::Int(25)); // Bob last
    }

    #[test]
    fn test_limit_offset() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        let plan = ExecutionPlan::new(
            PlanNode::Scan {
                table: "users".to_string(),
                columns: vec![],
                filter: None,
            }
            .limit(Some(2), Some(1)),
        );

        let result = executor.execute(&plan, &context).unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_aggregation() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        let plan = ExecutionPlan::new(PlanNode::Aggregate {
            input: Box::new(PlanNode::Scan {
                table: "users".to_string(),
                columns: vec![],
                filter: None,
            }),
            group_by: vec![],
            aggregates: vec![
                ("count".to_string(), "COUNT".to_string(), "id".to_string()),
                ("avg_age".to_string(), "AVG".to_string(), "age".to_string()),
            ],
        });

        let result = executor.execute(&plan, &context).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].values[0], Value::Int(3)); // count
        assert_eq!(result.rows[0].values[1], Value::Float(30.0)); // avg age
    }

    #[test]
    fn test_inner_join() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        let condition = Expression::Binary {
            left: Box::new(Expression::Column("id".to_string())),
            op: Operator::Eq,
            right: Box::new(Expression::Column("user_id".to_string())),
        };

        let plan = ExecutionPlan::new(PlanNode::Join {
            left: Box::new(PlanNode::Scan {
                table: "users".to_string(),
                columns: vec![],
                filter: None,
            }),
            right: Box::new(PlanNode::Scan {
                table: "orders".to_string(),
                columns: vec![],
                filter: None,
            }),
            join_type: crate::ast::JoinType::Inner,
            condition: Some(condition),
        });

        let result = executor.execute(&plan, &context).unwrap();
        assert_eq!(result.rows.len(), 3); // 2 orders for Alice, 1 for Bob
    }

    #[test]
    fn test_scalar_functions() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        // Test UPPER/LOWER
        let row = RowData::new(
            vec!["name".to_string()],
            vec![Value::String("Hello".to_string())],
        );
        let expr = Expression::Function {
            name: "UPPER".to_string(),
            args: vec![Expression::Column("name".to_string())],
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::String("HELLO".to_string()));

        // Test TRIM
        let row = RowData::new(
            vec!["s".to_string()],
            vec![Value::String("  hello  ".to_string())],
        );
        let expr = Expression::Function {
            name: "TRIM".to_string(),
            args: vec![Expression::Column("s".to_string())],
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::String("hello".to_string()));

        // Test REPLACE
        let expr = Expression::Function {
            name: "REPLACE".to_string(),
            args: vec![
                Expression::Literal(Value::String("hello world".to_string())),
                Expression::Literal(Value::String("world".to_string())),
                Expression::Literal(Value::String("rust".to_string())),
            ],
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::String("hello rust".to_string()));

        // Test ROUND
        let expr = Expression::Function {
            name: "ROUND".to_string(),
            args: vec![
                Expression::Literal(Value::Float(3.14159)),
                Expression::Literal(Value::Int(2)),
            ],
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::Float(3.14));

        // Test SQRT
        let expr = Expression::Function {
            name: "SQRT".to_string(),
            args: vec![Expression::Literal(Value::Float(16.0))],
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::Float(4.0));

        // Test POWER
        let expr = Expression::Function {
            name: "POWER".to_string(),
            args: vec![
                Expression::Literal(Value::Float(2.0)),
                Expression::Literal(Value::Float(3.0)),
            ],
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::Float(8.0));

        // Test CONCAT
        let expr = Expression::Function {
            name: "CONCAT".to_string(),
            args: vec![
                Expression::Literal(Value::String("Hello".to_string())),
                Expression::Literal(Value::String(" ".to_string())),
                Expression::Literal(Value::String("World".to_string())),
            ],
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::String("Hello World".to_string()));

        // Test CEIL/FLOOR
        let expr = Expression::Function {
            name: "CEIL".to_string(),
            args: vec![Expression::Literal(Value::Float(3.2))],
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::Float(4.0));

        let expr = Expression::Function {
            name: "FLOOR".to_string(),
            args: vec![Expression::Literal(Value::Float(3.8))],
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::Float(3.0));

        // Test RANDOM returns 0..1
        let expr = Expression::Function {
            name: "RANDOM".to_string(),
            args: vec![],
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        if let Value::Float(f) = result {
            assert!(f >= 0.0 && f < 1.0);
        } else {
            panic!("RANDOM should return Float");
        }
    }

    #[test]
    fn test_cast_int_to_string() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();
        let row = RowData::new(vec![], vec![]);

        let expr = Expression::Cast {
            expr: Box::new(Expression::Literal(Value::Int(42))),
            target_type: "TEXT".to_string(),
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::String("42".to_string()));
    }

    #[test]
    fn test_cast_string_to_int() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();
        let row = RowData::new(vec![], vec![]);

        let expr = Expression::Cast {
            expr: Box::new(Expression::Literal(Value::String("123".to_string()))),
            target_type: "INT".to_string(),
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::Int(123));
    }

    #[test]
    fn test_cast_float_to_int() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();
        let row = RowData::new(vec![], vec![]);

        let expr = Expression::Cast {
            expr: Box::new(Expression::Literal(Value::Float(3.7))),
            target_type: "INT".to_string(),
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::Int(3));
    }

    #[test]
    fn test_cast_bool_to_int() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();
        let row = RowData::new(vec![], vec![]);

        let expr = Expression::Cast {
            expr: Box::new(Expression::Literal(Value::Bool(true))),
            target_type: "INT".to_string(),
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::Int(1));
    }

    #[test]
    fn test_cast_in_expression() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();
        let row = RowData::new(vec![], vec![]);

        // CAST('42' AS INT) + 8 = 50
        let expr = Expression::Binary {
            left: Box::new(Expression::Cast {
                expr: Box::new(Expression::Literal(Value::String("42".to_string()))),
                target_type: "INT".to_string(),
            }),
            op: crate::ast::Operator::Add,
            right: Box::new(Expression::Literal(Value::Int(8))),
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::Int(50));
    }

    #[test]
    fn test_union_plan() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        let plan = ExecutionPlan::new(PlanNode::SetOp {
            left: Box::new(PlanNode::Scan {
                table: "users".to_string(),
                columns: vec![],
                filter: None,
            }),
            right: Box::new(PlanNode::Scan {
                table: "users".to_string(),
                columns: vec![],
                filter: None,
            }),
            op: SetOperationType::UnionAll,
        });

        let result = executor.execute(&plan, &context).unwrap();
        assert_eq!(result.rows.len(), 6); // 3 + 3 with UNION ALL
    }

    #[test]
    fn test_union_dedup() {
        let storage = setup_test_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        let plan = ExecutionPlan::new(PlanNode::SetOp {
            left: Box::new(PlanNode::Scan {
                table: "users".to_string(),
                columns: vec![],
                filter: None,
            }),
            right: Box::new(PlanNode::Scan {
                table: "users".to_string(),
                columns: vec![],
                filter: None,
            }),
            op: SetOperationType::Union,
        });

        let result = executor.execute(&plan, &context).unwrap();
        assert_eq!(result.rows.len(), 3); // Deduped to 3
    }

    fn setup_subquery_data() -> Arc<MemoryTableStorage> {
        let storage = Arc::new(MemoryTableStorage::new());

        storage.create_table(
            "products",
            vec!["id".to_string(), "name".to_string(), "price".to_string()],
        );
        for (id, name, price) in &[
            (1i64, "Widget", 10i64),
            (2, "Gadget", 25),
            (3, "Doohickey", 5),
        ] {
            storage.insert_row(
                "products",
                vec![
                    Value::Int(*id),
                    Value::String(name.to_string()),
                    Value::Int(*price),
                ],
            );
        }

        storage.create_table(
            "orders",
            vec![
                "id".to_string(),
                "product_id".to_string(),
                "qty".to_string(),
            ],
        );
        for (id, pid, qty) in &[(1i64, 1i64, 100i64), (2, 2, 50), (3, 1, 200)] {
            storage.insert_row(
                "orders",
                vec![Value::Int(*id), Value::Int(*pid), Value::Int(*qty)],
            );
        }

        storage
    }

    #[test]
    fn test_subquery_scalar() {
        use crate::ast::Query;
        use std::collections::HashMap as HM;

        let storage = setup_subquery_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        // Simulate: SELECT * FROM products WHERE price > (SELECT 15 LIMIT 1)
        // Subquery returns literal 15 via derived_columns
        let mut derived = HM::new();
        derived.insert("col_0".to_string(), Expression::Literal(Value::Int(15)));
        let subquery = Query {
            query_type: crate::ast::QueryType::Select,
            source: Some("products".to_string()),
            columns: vec!["col_0".to_string()],
            filter: None,
            order_by: vec![],
            group_by: vec![],
            having: None,
            limit: Some(1),
            offset: None,
            joins: vec![],
            values: vec![],
            returning: vec![],
            ctes: vec![],
            derived_columns: derived,
            distinct: false, source_alias: None,
        };

        // Test scalar subquery evaluation
        let row = RowData::new(
            vec!["id".to_string(), "name".to_string(), "price".to_string()],
            vec![
                Value::Int(2),
                Value::String("Gadget".to_string()),
                Value::Int(25),
            ],
        );

        let result = executor
            .eval_expression(&Expression::Subquery(Box::new(subquery)), &row, &context)
            .unwrap();
        assert_eq!(result, Value::Int(15));
    }

    #[test]
    fn test_subquery_exists_eval() {
        use crate::ast::Query;

        let storage = setup_subquery_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        // EXISTS (SELECT * FROM orders WHERE product_id = 1)
        let subquery = Query {
            query_type: crate::ast::QueryType::Select,
            source: Some("orders".to_string()),
            columns: vec!["*".to_string()],
            filter: Some(Expression::Binary {
                left: Box::new(Expression::Column("product_id".to_string())),
                op: Operator::Eq,
                right: Box::new(Expression::Literal(Value::Int(1))),
            }),
            order_by: vec![],
            group_by: vec![],
            having: None,
            limit: None,
            offset: None,
            joins: vec![],
            values: vec![],
            returning: vec![],
            ctes: vec![],
            derived_columns: std::collections::HashMap::new(),
            distinct: false, source_alias: None,
        };

        let row = RowData::new(vec![], vec![]);
        let result = executor
            .eval_expression(&Expression::Exists(Box::new(subquery)), &row, &context)
            .unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_subquery_exists_empty() {
        use crate::ast::Query;

        let storage = setup_subquery_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        // EXISTS (SELECT * FROM orders WHERE product_id = 999)
        let subquery = Query {
            query_type: crate::ast::QueryType::Select,
            source: Some("orders".to_string()),
            columns: vec!["*".to_string()],
            filter: Some(Expression::Binary {
                left: Box::new(Expression::Column("product_id".to_string())),
                op: Operator::Eq,
                right: Box::new(Expression::Literal(Value::Int(999))),
            }),
            order_by: vec![],
            group_by: vec![],
            having: None,
            limit: None,
            offset: None,
            joins: vec![],
            values: vec![],
            returning: vec![],
            ctes: vec![],
            derived_columns: std::collections::HashMap::new(),
            distinct: false, source_alias: None,
        };

        let row = RowData::new(vec![], vec![]);
        let result = executor
            .eval_expression(&Expression::Exists(Box::new(subquery)), &row, &context)
            .unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_subquery_in_with_subquery_item() {
        use crate::ast::Query;

        let storage = setup_subquery_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        // id IN (SELECT product_id FROM orders)
        let subquery = Query {
            query_type: crate::ast::QueryType::Select,
            source: Some("orders".to_string()),
            columns: vec!["product_id".to_string()],
            filter: None,
            order_by: vec![],
            group_by: vec![],
            having: None,
            limit: None,
            offset: None,
            joins: vec![],
            values: vec![],
            returning: vec![],
            ctes: vec![],
            derived_columns: std::collections::HashMap::new(),
            distinct: false, source_alias: None,
        };

        // Product 1 should be IN the subquery results
        let row = RowData::new(vec!["id".to_string()], vec![Value::Int(1)]);
        let in_expr = Expression::In {
            expr: Box::new(Expression::Column("id".to_string())),
            list: vec![Expression::Subquery(Box::new(subquery.clone()))],
            negated: false,
        };
        let result = executor.eval_expression(&in_expr, &row, &context).unwrap();
        assert_eq!(result, Value::Bool(true));

        // Product 3 should NOT be IN the subquery results
        let row3 = RowData::new(vec!["id".to_string()], vec![Value::Int(3)]);
        let subquery2 = Query {
            query_type: crate::ast::QueryType::Select,
            source: Some("orders".to_string()),
            columns: vec!["product_id".to_string()],
            filter: None,
            order_by: vec![],
            group_by: vec![],
            having: None,
            limit: None,
            offset: None,
            joins: vec![],
            values: vec![],
            returning: vec![],
            ctes: vec![],
            derived_columns: std::collections::HashMap::new(),
            distinct: false, source_alias: None,
        };
        let in_expr2 = Expression::In {
            expr: Box::new(Expression::Column("id".to_string())),
            list: vec![Expression::Subquery(Box::new(subquery2))],
            negated: false,
        };
        let result2 = executor
            .eval_expression(&in_expr2, &row3, &context)
            .unwrap();
        assert_eq!(result2, Value::Bool(false));
    }

    #[test]
    fn test_subquery_correlated() {
        use crate::ast::Query;

        let storage = setup_subquery_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        // Correlated: EXISTS (SELECT * FROM orders WHERE orders.product_id = products.id)
        // The outer row provides products.id, inner scan provides orders.product_id
        let subquery = Query {
            query_type: crate::ast::QueryType::Select,
            source: Some("orders".to_string()),
            columns: vec!["*".to_string()],
            filter: Some(Expression::Binary {
                left: Box::new(Expression::Column("product_id".to_string())),
                op: Operator::Eq,
                right: Box::new(Expression::Column("id".to_string())),
            }),
            order_by: vec![],
            group_by: vec![],
            having: None,
            limit: None,
            offset: None,
            joins: vec![],
            values: vec![],
            returning: vec![],
            ctes: vec![],
            derived_columns: std::collections::HashMap::new(),
            distinct: false, source_alias: None,
        };

        // Product 1 has orders
        let row1 = RowData::new(
            vec!["id".to_string(), "name".to_string(), "price".to_string()],
            vec![
                Value::Int(1),
                Value::String("Widget".to_string()),
                Value::Int(10),
            ],
        );
        let result = executor
            .eval_expression(
                &Expression::Exists(Box::new(subquery.clone())),
                &row1,
                &context,
            )
            .unwrap();
        assert_eq!(result, Value::Bool(true));

        // Product 3 has no orders
        let row3 = RowData::new(
            vec!["id".to_string(), "name".to_string(), "price".to_string()],
            vec![
                Value::Int(3),
                Value::String("Doohickey".to_string()),
                Value::Int(5),
            ],
        );
        let result3 = executor
            .eval_expression(&Expression::Exists(Box::new(subquery)), &row3, &context)
            .unwrap();
        assert_eq!(result3, Value::Bool(false));
    }

    #[test]
    fn test_subquery_null_comparison() {
        let storage = setup_subquery_data();
        let executor = StorageExecutor::new(storage);
        let context = QueryContext::default();

        // price > NULL should yield NULL (treated as false)
        let row = RowData::new(vec!["price".to_string()], vec![Value::Int(25)]);
        let expr = Expression::Binary {
            left: Box::new(Expression::Literal(Value::Int(25))),
            op: Operator::Gt,
            right: Box::new(Expression::Literal(Value::Null)),
        };
        let result = executor.eval_expression(&expr, &row, &context).unwrap();
        assert_eq!(result, Value::Null);
    }
}
