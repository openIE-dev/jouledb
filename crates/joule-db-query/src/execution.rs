//! Query Execution Engine
//!
//! Provides query execution with optional storage backend integration.
//!
//! ## Usage
//!
//! ### Without Storage (Planning Only)
//! ```ignore
//! let engine = QueryEngine::new();
//! let plan = engine.prepare(&query)?; // Get execution plan
//! ```
//!
//! ### With Storage (Full Execution)
//! ```ignore
//! let storage = Arc::new(MyTableStorage::new());
//! let engine = QueryEngine::with_storage(storage);
//! let result = engine.execute(&query, &context)?; // Actually executes
//! ```

#[cfg(feature = "parallel")]
#[path = "execution/parallel.rs"]
mod execution_parallel;
#[cfg(feature = "parallel")]
#[path = "execution/timeout_integration.rs"]
mod execution_timeout_integration;

#[cfg(feature = "parallel")]
pub use self::execution_parallel::{ParallelConfig, ParallelExecutor};
#[cfg(feature = "parallel")]
pub use self::execution_timeout_integration::{execute_with_checkpoints, execute_with_timeout};

use crate::ast::{Expression, Operator, Query, QueryType, Value};
use crate::error::{QueryError, QueryResult};
use crate::executor::{RowData, TableStorage};
use crate::planner::{PlanNode, QueryPlanner};
use crate::sql::SetOperationType;
use chrono::{Datelike, Timelike};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Query execution context
#[derive(Debug, Clone)]
pub struct QueryContext {
    /// Named parameters
    pub parameters: HashMap<String, Value>,
    /// Positional parameters
    pub positional_params: Vec<Value>,
    /// Query timeout
    pub timeout: Option<Duration>,
    /// Maximum rows to return
    pub max_rows: Option<usize>,
    /// Enable explain mode
    pub explain: bool,
    /// Read-only mode
    pub read_only: bool,
    /// Transaction ID
    pub transaction_id: Option<String>,
    /// Current subquery nesting depth (prevents stack overflow)
    pub subquery_depth: u32,
}

/// Maximum allowed subquery nesting depth.
pub const MAX_SUBQUERY_DEPTH: u32 = 64;

impl Default for QueryContext {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            positional_params: Vec::new(),
            timeout: Some(Duration::from_secs(30)),
            max_rows: Some(10000),
            explain: false,
            read_only: false,
            transaction_id: None,
            subquery_depth: 0,
        }
    }
}

impl QueryContext {
    /// Create new context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add named parameter
    pub fn with_param(mut self, name: &str, value: Value) -> Self {
        self.parameters.insert(name.to_string(), value);
        self
    }

    /// Add positional parameter
    pub fn with_positional(mut self, value: Value) -> Self {
        self.positional_params.push(value);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set max rows
    pub fn with_max_rows(mut self, max: usize) -> Self {
        self.max_rows = Some(max);
        self
    }

    /// Enable explain mode
    pub fn explain(mut self) -> Self {
        self.explain = true;
        self
    }

    /// Set read-only mode
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }
}

/// Execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Root plan node
    pub root: PlanNode,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Estimated rows
    pub estimated_rows: usize,
    /// Plan warnings
    pub warnings: Vec<String>,
}

impl ExecutionPlan {
    /// Create new plan
    pub fn new(root: PlanNode) -> Self {
        Self {
            root,
            estimated_cost: 0.0,
            estimated_rows: 0,
            warnings: Vec::new(),
        }
    }

    /// Add warning
    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }
}

/// Query result row
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    /// Column values
    pub values: Vec<Value>,
}

impl Row {
    /// Create new row
    pub fn new(values: Vec<Value>) -> Self {
        Self { values }
    }

    /// Get value at index
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.values.get(index)
    }
}

/// Query result set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultSet {
    /// Column names
    pub columns: Vec<String>,
    /// Rows
    pub rows: Vec<Row>,
    /// Affected rows (for INSERT/UPDATE/DELETE)
    pub affected_rows: usize,
    /// Execution time
    pub execution_time_ms: u64,
    /// Whether result was truncated
    pub truncated: bool,
}

impl ResultSet {
    /// Create empty result set
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            affected_rows: 0,
            execution_time_ms: 0,
            truncated: false,
        }
    }

    /// Create result set with columns
    pub fn with_columns(columns: Vec<String>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
            affected_rows: 0,
            execution_time_ms: 0,
            truncated: false,
        }
    }

    /// Add row
    pub fn add_row(&mut self, row: Row) {
        self.rows.push(row);
    }

    /// Get number of rows
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Get row by index
    pub fn row(&self, index: usize) -> Option<&Row> {
        self.rows.get(index)
    }

    /// Iterate over rows
    pub fn iter(&self) -> impl Iterator<Item = &Row> {
        self.rows.iter()
    }
}

/// Query engine trait
pub trait QueryExecutor: Send + Sync {
    /// Execute a query
    fn execute(&self, query: &Query, context: &QueryContext) -> QueryResult<ResultSet>;

    /// Prepare a query (returns plan)
    fn prepare(&self, query: &Query) -> QueryResult<ExecutionPlan>;

    /// Execute a raw query string
    fn execute_raw(&self, query: &str, context: &QueryContext) -> QueryResult<ResultSet>;
}

/// Query engine with optional storage backend
pub struct QueryEngine<S: TableStorage + 'static = NoOpStorage> {
    planner: QueryPlanner,
    timeout: Duration,
    storage: Option<Arc<S>>,
}

/// No-op storage implementation for planning-only mode
pub struct NoOpStorage;

impl TableStorage for NoOpStorage {
    fn scan(&self, table: &str) -> QueryResult<Vec<RowData>> {
        Err(QueryError::ExecutionError(format!(
            "No storage backend configured. Cannot scan table '{}'. Use QueryEngine::with_storage() to provide a storage backend.",
            table
        )))
    }

    fn columns(&self, table: &str) -> QueryResult<Vec<String>> {
        Err(QueryError::ExecutionError(format!(
            "No storage backend configured. Cannot get columns for table '{}'.",
            table
        )))
    }

    fn insert(&self, table: &str, _row: &RowData) -> QueryResult<()> {
        Err(QueryError::ExecutionError(format!(
            "No storage backend configured. Cannot insert into table '{}'.",
            table
        )))
    }

    fn update(
        &self,
        table: &str,
        _assignments: &HashMap<String, Value>,
        _predicate: Option<&Expression>,
    ) -> QueryResult<usize> {
        Err(QueryError::ExecutionError(format!(
            "No storage backend configured. Cannot update table '{}'.",
            table
        )))
    }

    fn delete(&self, table: &str, _predicate: Option<&Expression>) -> QueryResult<usize> {
        Err(QueryError::ExecutionError(format!(
            "No storage backend configured. Cannot delete from table '{}'.",
            table
        )))
    }

    fn table_exists(&self, _table: &str) -> QueryResult<bool> {
        Ok(false)
    }

    fn index_scan(
        &self,
        table: &str,
        _index: &str,
        _predicate: &Expression,
    ) -> QueryResult<Vec<RowData>> {
        Err(QueryError::ExecutionError(format!(
            "No storage backend configured. Cannot index scan table '{}'.",
            table
        )))
    }
}

impl QueryEngine<NoOpStorage> {
    /// Create new query engine without storage (planning only)
    pub fn new() -> Self {
        Self {
            planner: QueryPlanner::new(),
            timeout: Duration::from_secs(30),
            storage: None,
        }
    }
}

impl<S: TableStorage + 'static> QueryEngine<S> {
    /// Create query engine with storage backend for actual execution
    pub fn with_storage(storage: Arc<S>) -> Self {
        Self {
            planner: QueryPlanner::new(),
            timeout: Duration::from_secs(30),
            storage: Some(storage),
        }
    }

    /// Set default timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Update planner statistics (e.g. from real table data)
    pub fn set_statistics(&mut self, stats: crate::planner::Statistics) {
        self.planner.statistics = stats;
    }

    /// Check if storage is available
    pub fn has_storage(&self) -> bool {
        self.storage.is_some()
    }

    /// Execute query
    pub fn execute(&self, query: &Query, context: &QueryContext) -> QueryResult<ResultSet> {
        let start = Instant::now();
        let timeout = context.timeout.unwrap_or(self.timeout);

        // Check read-only mode
        if context.read_only {
            match query.query_type {
                QueryType::Select | QueryType::Traverse | QueryType::Aggregate => {}
                _ => {
                    return Err(QueryError::Unsupported(
                        "Write operations not allowed in read-only mode".to_string(),
                    ));
                }
            }
        }

        // Create execution plan
        let plan = self.planner.plan(query)?;

        // Check timeout
        if start.elapsed() > timeout {
            return Err(QueryError::Timeout);
        }

        // If explain mode, return plan as result
        if context.explain {
            return self.explain_plan(&plan);
        }

        // Execute plan
        let result = self.execute_plan(&plan, context)?;

        // Apply max rows limit
        let mut result = result;
        if let Some(max) = context.max_rows {
            if result.rows.len() > max {
                result.rows.truncate(max);
                result.truncated = true;
            }
        }

        result.execution_time_ms = start.elapsed().as_millis() as u64;
        Ok(result)
    }

    /// Prepare query (get plan without executing)
    pub fn prepare(&self, query: &Query) -> QueryResult<ExecutionPlan> {
        self.planner.plan(query)
    }

    fn explain_plan(&self, plan: &ExecutionPlan) -> QueryResult<ResultSet> {
        let mut result = ResultSet::with_columns(vec![
            "operation".to_string(),
            "estimated_cost".to_string(),
            "estimated_rows".to_string(),
            "cascade_tier".to_string(),
        ]);

        result.add_row(Row::new(vec![
            Value::String(format!("{:?}", plan.root)),
            Value::Float(plan.estimated_cost),
            Value::Int(plan.estimated_rows as i64),
            Value::String(plan.root.cascade_tier().to_string()),
        ]));

        Ok(result)
    }

    fn execute_plan(&self, plan: &ExecutionPlan, context: &QueryContext) -> QueryResult<ResultSet> {
        // Execute plan nodes recursively
        self.execute_node(&plan.root, context)
    }

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
            } => self.execute_index_scan(table, index, columns, filter, context),
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
                if let Some(off) = offset {
                    if *off < result.rows.len() {
                        result.rows = result.rows[*off..].to_vec();
                    } else {
                        result.rows.clear();
                    }
                }
                if let Some(lim) = limit {
                    result.rows.truncate(*lim);
                }
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
                let result = self.execute_node(input, context)?;
                self.execute_window(result, window_functions, context)
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

                let row_key = |row: &Row| -> String { format!("{:?}", row) };

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
                // Return empty result when reached through the generic executor.
                Ok(ResultSet::empty())
            }
            PlanNode::Empty => Ok(ResultSet::empty()),
        }
    }

    /// Execute a table scan against storage
    fn execute_scan(
        &self,
        table: &str,
        columns: &[String],
        filter: Option<&Expression>,
        context: &QueryContext,
    ) -> QueryResult<ResultSet> {
        // Get storage backend (use NoOpStorage if none configured)
        let storage: &dyn TableStorage = match &self.storage {
            Some(s) => s.as_ref(),
            None => return Err(QueryError::ExecutionError(
                "No storage backend configured. Use QueryEngine::with_storage() to provide one."
                    .to_string(),
            )),
        };

        // Get all rows from storage
        let rows = storage.scan(table)?;
        let table_columns = storage.columns(table)?;

        // Determine which columns to include
        // If columns contains "*" or is empty, use all table columns
        let result_columns: Vec<String> = if columns.is_empty() || columns.iter().any(|c| c == "*")
        {
            table_columns.clone()
        } else {
            columns.to_vec()
        };

        let use_all_columns = columns.is_empty() || columns.iter().any(|c| c == "*");
        let mut result = ResultSet::with_columns(result_columns.clone());

        for row_data in rows {
            // Apply filter if present
            if let Some(pred) = filter {
                if !self.eval_predicate(pred, &row_data, context)? {
                    continue;
                }
            }

            // Project columns
            let values: Vec<Value> = if use_all_columns {
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

    /// Execute an index scan
    fn execute_index_scan(
        &self,
        table: &str,
        index: &str,
        columns: &[String],
        filter: &Expression,
        _context: &QueryContext,
    ) -> QueryResult<ResultSet> {
        let storage: &dyn TableStorage = match &self.storage {
            Some(s) => s.as_ref(),
            None => {
                return Err(QueryError::ExecutionError(
                    "No storage backend configured.".to_string(),
                ));
            }
        };

        let rows = storage.index_scan(table, index, filter)?;
        let table_columns = storage.columns(table)?;

        let result_columns: Vec<String> = if columns.is_empty() || columns.iter().any(|c| c == "*")
        {
            table_columns.clone()
        } else {
            columns.to_vec()
        };

        let use_all_columns = columns.is_empty() || columns.iter().any(|c| c == "*");
        let mut result = ResultSet::with_columns(result_columns.clone());

        for row_data in rows {
            // Note: We assume index_scan already filtered rows based on the predicate

            let values: Vec<Value> = if use_all_columns {
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
        let columns = result.columns.clone();
        let filtered_rows: Vec<Row> = result
            .rows
            .into_iter()
            .filter(|row| {
                let row_data = RowData::new(columns.clone(), row.values.clone());
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
                    (Value::Float(a), crate::ast::Operator::Mul, Value::Float(b)) => {
                        Value::Float(a * b)
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

    /// Apply aggregation
    fn apply_aggregation(
        &self,
        result: ResultSet,
        group_by: &[Expression],
        aggregates: &[(String, String, String)], // (alias, function, column)
    ) -> QueryResult<ResultSet> {
        if group_by.is_empty() && aggregates.is_empty() {
            return Ok(result);
        }

        // Group rows
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
                for left_row in &left.rows {
                    for right_row in &right.rows {
                        result.add_row(self.combine_rows(left_row, right_row));
                    }
                }
            }
        }

        Ok(result)
    }

    /// Execute window function operations
    fn execute_window(
        &self,
        mut result: ResultSet,
        window_functions: &[crate::planner::WindowFunctionDef],
        _context: &QueryContext,
    ) -> QueryResult<ResultSet> {
        // For each window function, add a new column with the computed values
        for wf in window_functions {
            // Add the new column
            result.columns.push(wf.alias.clone());

            // Partition the rows if needed
            let partition_cols: Vec<&str> = wf
                .window
                .partition_by
                .iter()
                .filter_map(|e| match e {
                    Expression::Column(name) => Some(name.as_str()),
                    _ => None,
                })
                .collect();

            // Group rows by partition
            let mut partitions: std::collections::HashMap<String, Vec<usize>> =
                std::collections::HashMap::new();

            for (idx, row) in result.rows.iter().enumerate() {
                let partition_key: String = partition_cols
                    .iter()
                    .map(|col| {
                        let col_idx = result.columns.iter().position(|c| c == *col);
                        col_idx
                            .and_then(|i| row.values.get(i))
                            .map(|v| format!("{:?}", v))
                            .unwrap_or_else(|| "NULL".to_string())
                    })
                    .collect::<Vec<_>>()
                    .join("|");
                partitions.entry(partition_key).or_default().push(idx);
            }

            // Compute window function for each partition
            let mut window_values: Vec<Value> = vec![Value::Null; result.rows.len()];

            for partition_indices in partitions.values() {
                // Sort within partition if needed
                let sorted_indices = if wf.window.order_by.is_empty() {
                    partition_indices.clone()
                } else {
                    // Sort indices based on ORDER BY columns
                    let mut indices = partition_indices.clone();
                    indices.sort_by(|&a, &b| {
                        for order in &wf.window.order_by {
                            let va = result
                                .rows
                                .get(a)
                                .map(|r| {
                                    self.eval_order_expr(&order.expr, &result.columns, &r.values)
                                })
                                .unwrap_or(Value::Null);
                            let vb = result
                                .rows
                                .get(b)
                                .map(|r| {
                                    self.eval_order_expr(&order.expr, &result.columns, &r.values)
                                })
                                .unwrap_or(Value::Null);
                            let nulls_first = order.nulls_first.unwrap_or(order.descending);
                            let cmp = match (&va, &vb) {
                                (Value::Null, Value::Null) => std::cmp::Ordering::Equal,
                                (Value::Null, _) => {
                                    if nulls_first {
                                        std::cmp::Ordering::Less
                                    } else {
                                        std::cmp::Ordering::Greater
                                    }
                                }
                                (_, Value::Null) => {
                                    if nulls_first {
                                        std::cmp::Ordering::Greater
                                    } else {
                                        std::cmp::Ordering::Less
                                    }
                                }
                                _ => Self::compare_values(&va, &vb),
                            };
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
                                if !self.order_by_values_equal(
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
                                if !self.order_by_values_equal(
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
                        let col_name = wf.args.first().and_then(|e| {
                            if let Expression::Column(name) = e {
                                Some(name.as_str())
                            } else {
                                None
                            }
                        });
                        let col_idx =
                            col_name.and_then(|name| result.columns.iter().position(|c| c == name));
                        for (i, &idx) in sorted_indices.iter().enumerate() {
                            if i >= offset {
                                let prev_idx = sorted_indices[i - offset];
                                window_values[idx] = col_idx
                                    .and_then(|ci| {
                                        result.rows.get(prev_idx).and_then(|r| r.values.get(ci))
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
                        let col_name = wf.args.first().and_then(|e| {
                            if let Expression::Column(name) = e {
                                Some(name.as_str())
                            } else {
                                None
                            }
                        });
                        let col_idx =
                            col_name.and_then(|name| result.columns.iter().position(|c| c == name));
                        for (i, &idx) in sorted_indices.iter().enumerate() {
                            if i + offset < sorted_indices.len() {
                                let next_idx = sorted_indices[i + offset];
                                window_values[idx] = col_idx
                                    .and_then(|ci| {
                                        result.rows.get(next_idx).and_then(|r| r.values.get(ci))
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
                                let (start, _end) = frame.frame_range(pos, sorted_indices.len());
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
                                let (_start, end) = frame.frame_range(pos, sorted_indices.len());
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
                                    result.rows.get(target_idx).and_then(|r| r.values.get(ci))
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
                                        result.rows.get(prev_idx).and_then(|r| r.values.get(ci))
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
                                        result.rows.get(prev_idx).and_then(|r| r.values.get(ci))
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
                                        result.rows.get(prev_idx).and_then(|r| r.values.get(ci))
                                    })
                                    .and_then(|v| match v {
                                        Value::Timestamp(t) => Some(*t as f64),
                                        Value::Int(t) => Some(*t as f64),
                                        _ => None,
                                    });
                                window_values[idx] = match (cur_val, prev_val, cur_ts, prev_ts) {
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
                                .and_then(|ci| result.rows.get(idx).and_then(|r| r.values.get(ci)))
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
                        let col_idx = wf.args.first().and_then(|e| {
                            if let Expression::Column(name) = e {
                                result.columns.iter().position(|c| c == name)
                            } else {
                                None
                            }
                        });
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
                        for (i, &idx) in sorted_indices.iter().enumerate() {
                            if let Some(v) = vals[i] {
                                window_values[idx] = Value::Float(v);
                            } else {
                                let prev = (0..i).rev().find_map(|j| vals[j].map(|v| (j, v)));
                                let next =
                                    ((i + 1)..vals.len()).find_map(|j| vals[j].map(|v| (j, v)));
                                window_values[idx] = match (prev, next) {
                                    (Some((pj, pv)), Some((nj, nv))) => {
                                        let frac = (i - pj) as f64 / (nj - pj) as f64;
                                        Value::Float(pv + frac * (nv - pv))
                                    }
                                    (Some((_, pv)), None) => Value::Float(pv),
                                    (None, Some((_, nv))) => Value::Float(nv),
                                    _ => Value::Null,
                                };
                            }
                        }
                    }
                    "SUM" | "AVG" | "COUNT" | "MIN" | "MAX" => {
                        if let Some(ref frame) = wf.window.frame {
                            // Frame-aware: compute aggregate per row over its frame window
                            for (pos, &idx) in sorted_indices.iter().enumerate() {
                                let (start, end) = frame.frame_range(pos, sorted_indices.len());
                                let end = end.min(sorted_indices.len().saturating_sub(1));
                                let frame_indices: Vec<usize> = if start <= end && end < sorted_indices.len() {
                                    sorted_indices[start..=end].to_vec()
                                } else {
                                    Vec::new()
                                };
                                let agg_value = self.compute_partition_aggregate(
                                    &result,
                                    &frame_indices,
                                    &wf.function,
                                    &wf.args,
                                )?;
                                window_values[idx] = agg_value;
                            }
                        } else {
                            // No frame: aggregate over entire partition
                            let agg_value = self.compute_partition_aggregate(
                                &result,
                                &sorted_indices,
                                &wf.function,
                                &wf.args,
                            )?;
                            for &idx in &sorted_indices {
                                window_values[idx] = agg_value.clone();
                            }
                        }
                    }
                    _ => {
                        // Unknown function - return NULL
                        for &idx in &sorted_indices {
                            window_values[idx] = Value::Null;
                        }
                    }
                }
            }

            // Add window values to each row
            for (row, val) in result.rows.iter_mut().zip(window_values.into_iter()) {
                row.values.push(val);
            }
        }

        Ok(result)
    }

    fn order_by_values_equal(
        &self,
        result: &ResultSet,
        idx_a: usize,
        idx_b: usize,
        order_by: &[crate::ast::OrderBy],
    ) -> bool {
        for ob in order_by {
            if let Expression::Column(name) = &ob.expr {
                let col_idx = result.columns.iter().position(|c| c == name);
                if let Some(ci) = col_idx {
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

    /// Compute an aggregate over a partition of rows
    fn compute_partition_aggregate(
        &self,
        result: &ResultSet,
        indices: &[usize],
        function: &str,
        args: &[Expression],
    ) -> QueryResult<Value> {
        // Get the column to aggregate (if specified)
        let col_name = args.first().and_then(|e| match e {
            Expression::Column(name) => Some(name.as_str()),
            _ => None,
        });

        let values: Vec<&Value> = indices
            .iter()
            .filter_map(|&idx| {
                let row = result.rows.get(idx)?;
                if let Some(col) = col_name {
                    let col_idx = result.columns.iter().position(|c| c == col)?;
                    row.values.get(col_idx)
                } else {
                    row.values.first()
                }
            })
            .collect();

        let is_wildcard = args.is_empty() || matches!(args.first(), Some(Expression::Wildcard));

        match function.to_uppercase().as_str() {
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
                let nums: Vec<f64> = values
                    .iter()
                    .filter_map(|v| match v {
                        Value::Int(i) => Some(*i as f64),
                        Value::Float(f) => Some(*f),
                        _ => None,
                    })
                    .collect();
                if nums.is_empty() {
                    Ok(Value::Null)
                } else {
                    Ok(Value::Float(nums.iter().sum()))
                }
            }
            "AVG" => {
                let nums: Vec<f64> = values
                    .iter()
                    .filter_map(|v| match v {
                        Value::Int(i) => Some(*i as f64),
                        Value::Float(f) => Some(*f),
                        _ => None,
                    })
                    .collect();
                if nums.is_empty() {
                    Ok(Value::Null)
                } else {
                    Ok(Value::Float(nums.iter().sum::<f64>() / nums.len() as f64))
                }
            }
            "MIN" => {
                let min = values
                    .iter()
                    .filter_map(|v| match v {
                        Value::Int(i) => Some(*i as f64),
                        Value::Float(f) => Some(*f),
                        _ => None,
                    })
                    .fold(f64::MAX, f64::min);
                if min == f64::MAX {
                    Ok(Value::Null)
                } else {
                    Ok(Value::Float(min))
                }
            }
            "MAX" => {
                let max = values
                    .iter()
                    .filter_map(|v| match v {
                        Value::Int(i) => Some(*i as f64),
                        Value::Float(f) => Some(*f),
                        _ => None,
                    })
                    .fold(f64::MIN, f64::max);
                if max == f64::MIN {
                    Ok(Value::Null)
                } else {
                    Ok(Value::Float(max))
                }
            }
            _ => Ok(Value::Null),
        }
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
            _ => Ok(true),
        }
    }

    fn execute_subquery_expr(
        &self,
        query: &Query,
        outer_row: &RowData,
        context: &QueryContext,
    ) -> QueryResult<Vec<RowData>> {
        if context.subquery_depth >= MAX_SUBQUERY_DEPTH {
            return Err(QueryError::ExecutionError(format!(
                "Subquery nesting depth exceeds maximum ({})",
                MAX_SUBQUERY_DEPTH
            )));
        }
        let deeper = QueryContext {
            subquery_depth: context.subquery_depth + 1,
            ..context.clone()
        };
        let context = &deeper;
        let storage = self.storage.as_ref().ok_or_else(|| {
            QueryError::Unsupported("Subqueries require a storage backend".to_string())
        })?;

        let table_name = match &query.source {
            Some(name) => name.as_str(),
            None => return Ok(Vec::new()),
        };

        let rows = storage.scan(table_name)?;

        let mut result = Vec::new();

        for inner_row in &rows {
            // Build combined row for correlated subqueries
            let mut combined_cols = outer_row.columns.clone();
            combined_cols.extend(inner_row.columns.iter().cloned());
            let mut combined_vals = outer_row.values.clone();
            combined_vals.extend(inner_row.values.iter().cloned());
            let combined = RowData::new(combined_cols, combined_vals);

            let matches = if let Some(ref filter) = query.filter {
                self.eval_predicate(filter, &combined, context)?
            } else {
                true
            };

            if matches {
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
                Ok(row.get(column).cloned().unwrap_or(Value::Null))
            }
            Expression::Binary { left, op, right } => {
                let lval = self.eval_expression(left, row, context)?;
                let rval = self.eval_expression(right, row, context)?;
                self.eval_binary_op(&lval, op, &rval)
            }
            Expression::Unary { op, expr: inner } => {
                let val = self.eval_expression(inner, row, context)?;
                self.eval_unary_op(op, &val)
            }
            Expression::IsNull {
                expr: inner,
                negated,
            } => {
                let val = self.eval_expression(inner, row, context)?;
                let is_null = matches!(val, Value::Null);
                Ok(Value::Bool(if *negated { !is_null } else { is_null }))
            }
            Expression::In {
                expr: inner,
                list,
                negated,
            } => {
                let val = self.eval_expression(inner, row, context)?;
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
                expr: inner,
                low,
                high,
                negated,
            } => {
                let val = self.eval_expression(inner, row, context)?;
                let low_val = self.eval_expression(low, row, context)?;
                let high_val = self.eval_expression(high, row, context)?;

                let in_range = Self::compare_values(&val, &low_val) != std::cmp::Ordering::Less
                    && Self::compare_values(&val, &high_val) != std::cmp::Ordering::Greater;

                Ok(Value::Bool(if *negated { !in_range } else { in_range }))
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
        let scanner = |table: &str| -> QueryResult<(Vec<String>, Vec<Vec<serde_json::Value>>)> {
            if let Some(storage) = &self.storage {
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
            } else {
                Err(crate::QueryError::ExecutionError(format!(
                    "Table '{}' not accessible",
                    table
                )))
            }
        };
        crate::functions::eval_scalar_function(name, args, Some(&scanner))
    }

    /// Convert a crate::ast::Value to serde_json::Value
    fn value_to_serde_json(val: &Value) -> serde_json::Value {
        match val {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(*b),
            Value::Int(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
            Value::Float(f) => serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Value::String(s) => {
                // Try to parse as JSON first; if it parses, use the parsed value
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                    parsed
                } else {
                    serde_json::Value::String(s.clone())
                }
            }
            Value::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(Self::value_to_serde_json).collect())
            }
            Value::Object(map) => {
                let obj: serde_json::Map<String, serde_json::Value> = map
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::value_to_serde_json(v)))
                    .collect();
                serde_json::Value::Object(obj)
            }
            Value::Bytes(b) => serde_json::Value::String(format!("{:?}", b)),
            Value::Timestamp(ts) => serde_json::Value::Number(serde_json::Number::from(*ts)),
            Value::Uuid(u) => serde_json::Value::String(u.clone()),
            Value::Vector(v) => serde_json::json!(v),
        }
    }

    /// Convert a serde_json::Value to crate::ast::Value
    fn serde_json_to_value(jv: serde_json::Value) -> Value {
        match jv {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else {
                    Value::Float(n.as_f64().unwrap_or(0.0))
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
                // Default percentile is 0.5 (median); percentile may be encoded as last value
                let percentile = if values.len() >= 2 {
                    match values.last() {
                        Some(Value::Float(f)) if *f > 0.0 && *f < 1.0 => *f,
                        _ => 0.5,
                    }
                } else {
                    0.5
                };
                let mut nums: Vec<f64> = values.iter().filter_map(|v| v.as_float()).collect();
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

impl Default for QueryEngine<NoOpStorage> {
    fn default() -> Self {
        Self::new()
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
            let count = rows.len();
            rows.clear();
            Ok(count)
        }

        fn table_exists(&self, table: &str) -> QueryResult<bool> {
            let tables = self.tables.read().unwrap();
            Ok(tables.contains_key(table))
        }
    }

    fn setup_test_storage() -> Arc<MemoryTableStorage> {
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
        storage
    }

    #[test]
    fn test_query_context() {
        let ctx = QueryContext::new()
            .with_param("id", Value::Int(1))
            .with_timeout(Duration::from_secs(60))
            .with_max_rows(100);

        assert_eq!(ctx.parameters.get("id"), Some(&Value::Int(1)));
        assert_eq!(ctx.timeout, Some(Duration::from_secs(60)));
        assert_eq!(ctx.max_rows, Some(100));
    }

    #[test]
    fn test_result_set() {
        let mut result = ResultSet::with_columns(vec!["id".to_string(), "name".to_string()]);
        result.add_row(Row::new(vec![
            Value::Int(1),
            Value::String("Alice".to_string()),
        ]));
        result.add_row(Row::new(vec![
            Value::Int(2),
            Value::String("Bob".to_string()),
        ]));

        assert_eq!(result.len(), 2);
        assert_eq!(result.columns, vec!["id", "name"]);
    }

    #[test]
    fn test_query_engine_without_storage() {
        let engine = QueryEngine::new();
        let query = Query::select("users").columns(vec!["id", "name"]).limit(10);
        let ctx = QueryContext::new();

        // Without storage, execution should fail
        let result = engine.execute(&query, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_query_engine_with_storage() {
        let storage = setup_test_storage();
        let engine = QueryEngine::with_storage(storage);
        let query = Query::select("users").columns(vec!["id", "name"]).limit(10);
        let ctx = QueryContext::new();

        let result = engine.execute(&query, &ctx).unwrap();
        assert_eq!(result.columns, vec!["id", "name"]);
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn test_query_engine_scan_all_columns() {
        let storage = setup_test_storage();
        let engine = QueryEngine::with_storage(storage);
        let query = Query::select("users");
        let ctx = QueryContext::new();

        let result = engine.execute(&query, &ctx).unwrap();
        assert_eq!(result.columns, vec!["id", "name", "age"]);
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn test_read_only_mode() {
        let storage = setup_test_storage();
        let engine = QueryEngine::with_storage(storage);
        let query = Query::insert("users");
        let ctx = QueryContext::new().read_only();

        let result = engine.execute(&query, &ctx);
        assert!(matches!(result, Err(QueryError::Unsupported(_))));
    }

    #[test]
    fn test_explain_mode() {
        let storage = setup_test_storage();
        let engine = QueryEngine::with_storage(storage);
        let query = Query::select("users");
        let ctx = QueryContext::new().explain();

        let result = engine.execute(&query, &ctx).unwrap();
        assert!(result.columns.contains(&"operation".to_string()));
        assert!(result.columns.contains(&"cascade_tier".to_string()));

        // For a SELECT * FROM users with no filter, the tier should be Extract.
        let tier_idx = result
            .columns
            .iter()
            .position(|c| c == "cascade_tier")
            .unwrap();
        let tier_cell = &result.rows[0].values[tier_idx];
        match tier_cell {
            Value::String(s) => assert_eq!(s, "Extract"),
            other => panic!("expected cascade_tier as String, got {other:?}"),
        }
    }

    #[test]
    fn test_query_with_filter() {
        let storage = setup_test_storage();
        let engine = QueryEngine::with_storage(storage);

        // Query with filter: age > 25
        let plan = ExecutionPlan::new(PlanNode::Scan {
            table: "users".to_string(),
            columns: vec![],
            filter: Some(Expression::Binary {
                left: Box::new(Expression::Column("age".to_string())),
                op: Operator::Gt,
                right: Box::new(Expression::Literal(Value::Int(25))),
            }),
        });

        let ctx = QueryContext::new();
        let result = engine.execute_plan(&plan, &ctx).unwrap();
        assert_eq!(result.rows.len(), 2); // Alice (30) and Charlie (35)
    }

    #[test]
    fn test_query_with_sort() {
        let storage = setup_test_storage();
        let engine = QueryEngine::with_storage(storage);

        // Query sorted by age descending
        let plan = ExecutionPlan::new(PlanNode::Sort {
            input: Box::new(PlanNode::Scan {
                table: "users".to_string(),
                columns: vec![],
                filter: None,
            }),
            order_by: vec![crate::ast::OrderBy {
                expr: Expression::Column("age".to_string()),
                descending: true,
                nulls_first: None,
            }],
        });

        let ctx = QueryContext::new();
        let result = engine.execute_plan(&plan, &ctx).unwrap();
        assert_eq!(result.rows[0].values[2], Value::Int(35)); // Charlie first
        assert_eq!(result.rows[2].values[2], Value::Int(25)); // Bob last
    }

    fn setup_ranked_storage() -> Arc<MemoryTableStorage> {
        let storage = Arc::new(MemoryTableStorage::new());
        storage.create_table(
            "scores",
            vec!["name".to_string(), "dept".to_string(), "score".to_string()],
        );
        storage.insert_row(
            "scores",
            vec![
                Value::String("Alice".to_string()),
                Value::String("A".to_string()),
                Value::Int(90),
            ],
        );
        storage.insert_row(
            "scores",
            vec![
                Value::String("Bob".to_string()),
                Value::String("A".to_string()),
                Value::Int(90),
            ],
        );
        storage.insert_row(
            "scores",
            vec![
                Value::String("Charlie".to_string()),
                Value::String("A".to_string()),
                Value::Int(80),
            ],
        );
        storage.insert_row(
            "scores",
            vec![
                Value::String("Diana".to_string()),
                Value::String("B".to_string()),
                Value::Int(95),
            ],
        );
        storage
    }

    #[test]
    fn test_window_row_number() {
        let storage = setup_ranked_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let plan = ExecutionPlan::new(PlanNode::Window {
            input: Box::new(PlanNode::Scan {
                table: "scores".to_string(),
                columns: vec![],
                filter: None,
            }),
            window_functions: vec![crate::planner::WindowFunctionDef {
                alias: "rn".to_string(),
                function: "ROW_NUMBER".to_string(),
                args: vec![],
                window: crate::ast::WindowSpec {
                    partition_by: vec![],
                    order_by: vec![crate::ast::OrderBy {
                        expr: Expression::Column("score".to_string()),
                        descending: true,
                        nulls_first: None,
                    }],
                    frame: None,
                },
            }],
        });

        let result = engine.execute_plan(&plan, &ctx).unwrap();
        let rn_idx = result.columns.iter().position(|c| c == "rn").unwrap();
        // Diana(95) should be rn=1
        let diana_row = result
            .rows
            .iter()
            .find(|r| r.values[0] == Value::String("Diana".to_string()))
            .unwrap();
        assert_eq!(diana_row.values[rn_idx], Value::Int(1));
    }

    #[test]
    fn test_window_rank_with_ties() {
        let storage = setup_ranked_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let plan = ExecutionPlan::new(PlanNode::Window {
            input: Box::new(PlanNode::Scan {
                table: "scores".to_string(),
                columns: vec![],
                filter: None,
            }),
            window_functions: vec![crate::planner::WindowFunctionDef {
                alias: "rnk".to_string(),
                function: "RANK".to_string(),
                args: vec![],
                window: crate::ast::WindowSpec {
                    partition_by: vec![],
                    order_by: vec![crate::ast::OrderBy {
                        expr: Expression::Column("score".to_string()),
                        descending: true,
                        nulls_first: None,
                    }],
                    frame: None,
                },
            }],
        });

        let result = engine.execute_plan(&plan, &ctx).unwrap();
        let rnk_idx = result.columns.iter().position(|c| c == "rnk").unwrap();

        // Collect ranks in sorted order
        let mut ranks: Vec<(i64, i64)> = result
            .rows
            .iter()
            .map(|r| {
                let score = if let Value::Int(s) = &r.values[2] {
                    *s
                } else {
                    0
                };
                let rank = if let Value::Int(r) = &r.values[rnk_idx] {
                    *r
                } else {
                    0
                };
                (score, rank)
            })
            .collect();
        ranks.sort_by(|a, b| b.0.cmp(&a.0)); // sort by score desc

        // 95 -> rank 1, 90 -> rank 2 (tied), 90 -> rank 2, 80 -> rank 4 (skip 3)
        assert_eq!(ranks[0], (95, 1));
        assert_eq!(ranks[1].1, 2); // first 90
        assert_eq!(ranks[2].1, 2); // second 90 (tie)
        assert_eq!(ranks[3], (80, 4)); // skipped rank 3
    }

    #[test]
    fn test_window_dense_rank() {
        let storage = setup_ranked_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let plan = ExecutionPlan::new(PlanNode::Window {
            input: Box::new(PlanNode::Scan {
                table: "scores".to_string(),
                columns: vec![],
                filter: None,
            }),
            window_functions: vec![crate::planner::WindowFunctionDef {
                alias: "drnk".to_string(),
                function: "DENSE_RANK".to_string(),
                args: vec![],
                window: crate::ast::WindowSpec {
                    partition_by: vec![],
                    order_by: vec![crate::ast::OrderBy {
                        expr: Expression::Column("score".to_string()),
                        descending: true,
                        nulls_first: None,
                    }],
                    frame: None,
                },
            }],
        });

        let result = engine.execute_plan(&plan, &ctx).unwrap();
        let idx = result.columns.iter().position(|c| c == "drnk").unwrap();

        let mut ranks: Vec<(i64, i64)> = result
            .rows
            .iter()
            .map(|r| {
                let score = if let Value::Int(s) = &r.values[2] {
                    *s
                } else {
                    0
                };
                let rank = if let Value::Int(r) = &r.values[idx] {
                    *r
                } else {
                    0
                };
                (score, rank)
            })
            .collect();
        ranks.sort_by(|a, b| b.0.cmp(&a.0));

        // 95 -> 1, 90 -> 2, 90 -> 2, 80 -> 3 (no gap)
        assert_eq!(ranks[0], (95, 1));
        assert_eq!(ranks[1].1, 2);
        assert_eq!(ranks[2].1, 2);
        assert_eq!(ranks[3], (80, 3)); // dense = no gap
    }

    #[test]
    fn test_window_lag() {
        let storage = setup_test_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let plan = ExecutionPlan::new(PlanNode::Window {
            input: Box::new(PlanNode::Scan {
                table: "users".to_string(),
                columns: vec![],
                filter: None,
            }),
            window_functions: vec![crate::planner::WindowFunctionDef {
                alias: "prev_age".to_string(),
                function: "LAG".to_string(),
                args: vec![Expression::Column("age".to_string())],
                window: crate::ast::WindowSpec {
                    partition_by: vec![],
                    order_by: vec![crate::ast::OrderBy {
                        expr: Expression::Column("age".to_string()),
                        descending: false,
                        nulls_first: None,
                    }],
                    frame: None,
                },
            }],
        });

        let result = engine.execute_plan(&plan, &ctx).unwrap();
        let lag_idx = result.columns.iter().position(|c| c == "prev_age").unwrap();

        // Sorted by age asc: Bob(25), Alice(30), Charlie(35)
        // LAG: Bob->NULL, Alice->25, Charlie->30
        let bob = result
            .rows
            .iter()
            .find(|r| r.values[1] == Value::String("Bob".to_string()))
            .unwrap();
        let alice = result
            .rows
            .iter()
            .find(|r| r.values[1] == Value::String("Alice".to_string()))
            .unwrap();
        let charlie = result
            .rows
            .iter()
            .find(|r| r.values[1] == Value::String("Charlie".to_string()))
            .unwrap();
        assert_eq!(bob.values[lag_idx], Value::Null);
        assert_eq!(alice.values[lag_idx], Value::Int(25));
        assert_eq!(charlie.values[lag_idx], Value::Int(30));
    }

    #[test]
    fn test_window_lead() {
        let storage = setup_test_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let plan = ExecutionPlan::new(PlanNode::Window {
            input: Box::new(PlanNode::Scan {
                table: "users".to_string(),
                columns: vec![],
                filter: None,
            }),
            window_functions: vec![crate::planner::WindowFunctionDef {
                alias: "next_age".to_string(),
                function: "LEAD".to_string(),
                args: vec![Expression::Column("age".to_string())],
                window: crate::ast::WindowSpec {
                    partition_by: vec![],
                    order_by: vec![crate::ast::OrderBy {
                        expr: Expression::Column("age".to_string()),
                        descending: false,
                        nulls_first: None,
                    }],
                    frame: None,
                },
            }],
        });

        let result = engine.execute_plan(&plan, &ctx).unwrap();
        let lead_idx = result.columns.iter().position(|c| c == "next_age").unwrap();

        // Sorted by age asc: Bob(25), Alice(30), Charlie(35)
        // LEAD: Bob->30, Alice->35, Charlie->NULL
        let bob = result
            .rows
            .iter()
            .find(|r| r.values[1] == Value::String("Bob".to_string()))
            .unwrap();
        let alice = result
            .rows
            .iter()
            .find(|r| r.values[1] == Value::String("Alice".to_string()))
            .unwrap();
        let charlie = result
            .rows
            .iter()
            .find(|r| r.values[1] == Value::String("Charlie".to_string()))
            .unwrap();
        assert_eq!(bob.values[lead_idx], Value::Int(30));
        assert_eq!(alice.values[lead_idx], Value::Int(35));
        assert_eq!(charlie.values[lead_idx], Value::Null);
    }

    #[test]
    fn test_window_sum_over_partition() {
        let storage = setup_ranked_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let plan = ExecutionPlan::new(PlanNode::Window {
            input: Box::new(PlanNode::Scan {
                table: "scores".to_string(),
                columns: vec![],
                filter: None,
            }),
            window_functions: vec![crate::planner::WindowFunctionDef {
                alias: "dept_total".to_string(),
                function: "SUM".to_string(),
                args: vec![Expression::Column("score".to_string())],
                window: crate::ast::WindowSpec {
                    partition_by: vec![Expression::Column("dept".to_string())],
                    order_by: vec![],
                    frame: None,
                },
            }],
        });

        let result = engine.execute_plan(&plan, &ctx).unwrap();
        let dt_idx = result
            .columns
            .iter()
            .position(|c| c == "dept_total")
            .unwrap();

        // Dept A total = 90+90+80 = 260.0, Dept B total = 95.0
        for row in &result.rows {
            if let Value::String(dept) = &row.values[1] {
                match dept.as_str() {
                    "A" => assert_eq!(row.values[dt_idx], Value::Float(260.0)),
                    "B" => assert_eq!(row.values[dt_idx], Value::Float(95.0)),
                    _ => panic!("Unexpected dept"),
                }
            }
        }
    }

    #[test]
    fn test_window_row_number_with_partition() {
        let storage = setup_ranked_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let plan = ExecutionPlan::new(PlanNode::Window {
            input: Box::new(PlanNode::Scan {
                table: "scores".to_string(),
                columns: vec![],
                filter: None,
            }),
            window_functions: vec![crate::planner::WindowFunctionDef {
                alias: "rn".to_string(),
                function: "ROW_NUMBER".to_string(),
                args: vec![],
                window: crate::ast::WindowSpec {
                    partition_by: vec![Expression::Column("dept".to_string())],
                    order_by: vec![crate::ast::OrderBy {
                        expr: Expression::Column("score".to_string()),
                        descending: true,
                        nulls_first: None,
                    }],
                    frame: None,
                },
            }],
        });

        let result = engine.execute_plan(&plan, &ctx).unwrap();
        let rn_idx = result.columns.iter().position(|c| c == "rn").unwrap();

        // Dept B has only Diana (rn=1)
        // Dept A has Alice(90), Bob(90), Charlie(80) -> rn 1,2,3
        let dept_b_rows: Vec<_> = result
            .rows
            .iter()
            .filter(|r| r.values[1] == Value::String("B".to_string()))
            .collect();
        assert_eq!(dept_b_rows.len(), 1);
        assert_eq!(dept_b_rows[0].values[rn_idx], Value::Int(1));

        // All dept A rows should have rn 1, 2, or 3
        let dept_a_rns: Vec<i64> = result
            .rows
            .iter()
            .filter(|r| r.values[1] == Value::String("A".to_string()))
            .map(|r| {
                if let Value::Int(n) = &r.values[rn_idx] {
                    *n
                } else {
                    0
                }
            })
            .collect();
        let mut sorted = dept_a_rns.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3]);
    }

    #[test]
    fn test_scalar_functions_execution() {
        let storage = setup_test_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        // Test eval_function directly via expression evaluation
        let row = RowData::new(vec![], vec![]);

        // SUBSTRING
        let expr = Expression::Function {
            name: "SUBSTRING".to_string(),
            args: vec![
                Expression::Literal(Value::String("Hello World".to_string())),
                Expression::Literal(Value::Int(7)),
                Expression::Literal(Value::Int(5)),
            ],
        };
        let result = engine.eval_expression(&expr, &row, &ctx).unwrap();
        assert_eq!(result, Value::String("World".to_string()));

        // LOG (natural log)
        let expr = Expression::Function {
            name: "LN".to_string(),
            args: vec![Expression::Literal(Value::Float(std::f64::consts::E))],
        };
        let result = engine.eval_expression(&expr, &row, &ctx).unwrap();
        if let Value::Float(f) = result {
            assert!((f - 1.0).abs() < 1e-10);
        } else {
            panic!("LN should return Float");
        }

        // NULLIF
        let expr = Expression::Function {
            name: "NULLIF".to_string(),
            args: vec![
                Expression::Literal(Value::Int(1)),
                Expression::Literal(Value::Int(1)),
            ],
        };
        let result = engine.eval_expression(&expr, &row, &ctx).unwrap();
        assert_eq!(result, Value::Null);

        let expr = Expression::Function {
            name: "NULLIF".to_string(),
            args: vec![
                Expression::Literal(Value::Int(1)),
                Expression::Literal(Value::Int(2)),
            ],
        };
        let result = engine.eval_expression(&expr, &row, &ctx).unwrap();
        assert_eq!(result, Value::Int(1));
    }

    #[test]
    fn test_cast_execution() {
        let storage = setup_test_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();
        let row = RowData::new(vec![], vec![]);

        // CAST('true' AS BOOL)
        let expr = Expression::Cast {
            expr: Box::new(Expression::Literal(Value::String("true".to_string()))),
            target_type: "BOOL".to_string(),
        };
        let result = engine.eval_expression(&expr, &row, &ctx).unwrap();
        assert_eq!(result, Value::Bool(true));

        // CAST(42 AS FLOAT)
        let expr = Expression::Cast {
            expr: Box::new(Expression::Literal(Value::Int(42))),
            target_type: "FLOAT".to_string(),
        };
        let result = engine.eval_expression(&expr, &row, &ctx).unwrap();
        assert_eq!(result, Value::Float(42.0));
    }

    #[test]
    fn test_union_execution() {
        let storage = setup_test_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        // UNION ALL: should concatenate
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
        let result = engine.execute_plan(&plan, &ctx).unwrap();
        assert_eq!(result.rows.len(), 6);

        // UNION (no ALL): should deduplicate
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
        let result = engine.execute_plan(&plan, &ctx).unwrap();
        assert_eq!(result.rows.len(), 3);
    }

    fn setup_subquery_storage() -> Arc<MemoryTableStorage> {
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
        let storage = setup_subquery_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let mut derived = HashMap::new();
        derived.insert("col_0".to_string(), Expression::Literal(Value::Int(15)));
        let subquery = Query {
            query_type: QueryType::Select,
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

        let row = RowData::new(vec!["price".to_string()], vec![Value::Int(25)]);
        let result = engine
            .eval_expression(&Expression::Subquery(Box::new(subquery)), &row, &ctx)
            .unwrap();
        assert_eq!(result, Value::Int(15));
    }

    #[test]
    fn test_subquery_exists_eval() {
        let storage = setup_subquery_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let subquery = Query {
            query_type: QueryType::Select,
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
            derived_columns: HashMap::new(),
            distinct: false, source_alias: None,
        };

        let row = RowData::new(vec![], vec![]);
        let result = engine
            .eval_expression(&Expression::Exists(Box::new(subquery)), &row, &ctx)
            .unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_subquery_exists_empty() {
        let storage = setup_subquery_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let subquery = Query {
            query_type: QueryType::Select,
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
            derived_columns: HashMap::new(),
            distinct: false, source_alias: None,
        };

        let row = RowData::new(vec![], vec![]);
        let result = engine
            .eval_expression(&Expression::Exists(Box::new(subquery)), &row, &ctx)
            .unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_subquery_in_with_subquery_item() {
        let storage = setup_subquery_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let subquery = Query {
            query_type: QueryType::Select,
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
            derived_columns: HashMap::new(),
            distinct: false, source_alias: None,
        };

        // Product 1 is in orders
        let row = RowData::new(vec!["id".to_string()], vec![Value::Int(1)]);
        let in_expr = Expression::In {
            expr: Box::new(Expression::Column("id".to_string())),
            list: vec![Expression::Subquery(Box::new(subquery.clone()))],
            negated: false,
        };
        let result = engine.eval_expression(&in_expr, &row, &ctx).unwrap();
        assert_eq!(result, Value::Bool(true));

        // Product 3 is NOT in orders
        let row3 = RowData::new(vec!["id".to_string()], vec![Value::Int(3)]);
        let in_expr2 = Expression::In {
            expr: Box::new(Expression::Column("id".to_string())),
            list: vec![Expression::Subquery(Box::new(subquery))],
            negated: false,
        };
        let result2 = engine.eval_expression(&in_expr2, &row3, &ctx).unwrap();
        assert_eq!(result2, Value::Bool(false));
    }

    #[test]
    fn test_subquery_correlated() {
        let storage = setup_subquery_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let subquery = Query {
            query_type: QueryType::Select,
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
            derived_columns: HashMap::new(),
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
        let result = engine
            .eval_expression(&Expression::Exists(Box::new(subquery.clone())), &row1, &ctx)
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
        let result3 = engine
            .eval_expression(&Expression::Exists(Box::new(subquery)), &row3, &ctx)
            .unwrap();
        assert_eq!(result3, Value::Bool(false));
    }

    #[test]
    fn test_subquery_requires_storage() {
        let engine = QueryEngine::<NoOpStorage>::new();
        let ctx = QueryContext::new();

        let subquery = Query {
            query_type: QueryType::Select,
            source: Some("products".to_string()),
            columns: vec!["*".to_string()],
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
            derived_columns: HashMap::new(),
            distinct: false, source_alias: None,
        };

        let row = RowData::new(vec![], vec![]);
        let result = engine.eval_expression(&Expression::Exists(Box::new(subquery)), &row, &ctx);
        assert!(result.is_err());
    }

    // ========================================================================
    // Phase 10: NTILE, FIRST_VALUE, LAST_VALUE, NTH_VALUE window functions
    // ========================================================================

    #[test]
    fn test_window_ntile() {
        let storage = setup_ranked_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        // 4 rows, NTILE(2) ordered by score desc: Diana(95)->1, Alice(90)->1, Bob(90)->2, Charlie(80)->2
        let plan = ExecutionPlan::new(PlanNode::Window {
            input: Box::new(PlanNode::Scan {
                table: "scores".to_string(),
                columns: vec![],
                filter: None,
            }),
            window_functions: vec![crate::planner::WindowFunctionDef {
                alias: "bucket".to_string(),
                function: "NTILE".to_string(),
                args: vec![Expression::Literal(Value::Int(2))],
                window: crate::ast::WindowSpec {
                    partition_by: vec![],
                    order_by: vec![crate::ast::OrderBy {
                        expr: Expression::Column("score".to_string()),
                        descending: true,
                        nulls_first: None,
                    }],
                    frame: None,
                },
            }],
        });

        let result = engine.execute_plan(&plan, &ctx).unwrap();
        let bucket_idx = result.columns.iter().position(|c| c == "bucket").unwrap();

        // Collect (score, bucket) pairs
        let mut pairs: Vec<(i64, i64)> = result
            .rows
            .iter()
            .map(|r| {
                let score = if let Value::Int(s) = &r.values[2] {
                    *s
                } else {
                    0
                };
                let bucket = if let Value::Int(b) = &r.values[bucket_idx] {
                    *b
                } else {
                    0
                };
                (score, bucket)
            })
            .collect();
        pairs.sort_by(|a, b| b.0.cmp(&a.0));

        // 4 rows / 2 buckets: first 2 in bucket 1, last 2 in bucket 2
        assert_eq!(pairs[0].1, 1); // score 95
        assert_eq!(pairs[1].1, 1); // score 90
        assert_eq!(pairs[2].1, 2); // score 90
        assert_eq!(pairs[3].1, 2); // score 80
    }

    #[test]
    fn test_window_first_last_value() {
        let storage = setup_ranked_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let plan = ExecutionPlan::new(PlanNode::Window {
            input: Box::new(PlanNode::Scan {
                table: "scores".to_string(),
                columns: vec![],
                filter: None,
            }),
            window_functions: vec![
                crate::planner::WindowFunctionDef {
                    alias: "first_name".to_string(),
                    function: "FIRST_VALUE".to_string(),
                    args: vec![Expression::Column("name".to_string())],
                    window: crate::ast::WindowSpec {
                        partition_by: vec![],
                        order_by: vec![crate::ast::OrderBy {
                            expr: Expression::Column("score".to_string()),
                            descending: true,
                            nulls_first: None,
                        }],
                        frame: None,
                    },
                },
                crate::planner::WindowFunctionDef {
                    alias: "last_name".to_string(),
                    function: "LAST_VALUE".to_string(),
                    args: vec![Expression::Column("name".to_string())],
                    window: crate::ast::WindowSpec {
                        partition_by: vec![],
                        order_by: vec![crate::ast::OrderBy {
                            expr: Expression::Column("score".to_string()),
                            descending: true,
                            nulls_first: None,
                        }],
                        frame: None,
                    },
                },
            ],
        });

        let result = engine.execute_plan(&plan, &ctx).unwrap();
        let first_idx = result
            .columns
            .iter()
            .position(|c| c == "first_name")
            .unwrap();
        let last_idx = result
            .columns
            .iter()
            .position(|c| c == "last_name")
            .unwrap();

        // Ordered by score desc: Diana(95), Alice(90), Bob(90), Charlie(80)
        // FIRST_VALUE = Diana (highest), LAST_VALUE = Charlie (lowest)
        for row in &result.rows {
            assert_eq!(row.values[first_idx], Value::String("Diana".to_string()));
            assert_eq!(row.values[last_idx], Value::String("Charlie".to_string()));
        }
    }

    #[test]
    fn test_window_nth_value() {
        let storage = setup_ranked_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        // NTH_VALUE(name, 2) ordered by score desc -> 2nd value is Alice or Bob (score 90)
        let plan = ExecutionPlan::new(PlanNode::Window {
            input: Box::new(PlanNode::Scan {
                table: "scores".to_string(),
                columns: vec![],
                filter: None,
            }),
            window_functions: vec![crate::planner::WindowFunctionDef {
                alias: "second_name".to_string(),
                function: "NTH_VALUE".to_string(),
                args: vec![
                    Expression::Column("name".to_string()),
                    Expression::Literal(Value::Int(2)),
                ],
                window: crate::ast::WindowSpec {
                    partition_by: vec![],
                    order_by: vec![crate::ast::OrderBy {
                        expr: Expression::Column("score".to_string()),
                        descending: true,
                        nulls_first: None,
                    }],
                    frame: None,
                },
            }],
        });

        let result = engine.execute_plan(&plan, &ctx).unwrap();
        let nth_idx = result
            .columns
            .iter()
            .position(|c| c == "second_name")
            .unwrap();

        // 2nd row sorted by score desc: Diana(95), then Alice or Bob (90)
        // All rows should have the same NTH_VALUE
        let val = &result.rows[0].values[nth_idx];
        assert!(
            *val == Value::String("Alice".to_string()) || *val == Value::String("Bob".to_string()),
            "Expected Alice or Bob as 2nd value, got {:?}",
            val
        );
        // All rows should have the same value
        for row in &result.rows {
            assert_eq!(row.values[nth_idx], *val);
        }
    }

    #[test]
    fn test_window_ntile_with_partition() {
        let storage = setup_ranked_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        // NTILE(2) PARTITION BY dept: Dept A has 3 rows -> buckets [1,1,2], Dept B has 1 row -> bucket [1]
        let plan = ExecutionPlan::new(PlanNode::Window {
            input: Box::new(PlanNode::Scan {
                table: "scores".to_string(),
                columns: vec![],
                filter: None,
            }),
            window_functions: vec![crate::planner::WindowFunctionDef {
                alias: "bucket".to_string(),
                function: "NTILE".to_string(),
                args: vec![Expression::Literal(Value::Int(2))],
                window: crate::ast::WindowSpec {
                    partition_by: vec![Expression::Column("dept".to_string())],
                    order_by: vec![crate::ast::OrderBy {
                        expr: Expression::Column("score".to_string()),
                        descending: true,
                        nulls_first: None,
                    }],
                    frame: None,
                },
            }],
        });

        let result = engine.execute_plan(&plan, &ctx).unwrap();
        let bucket_idx = result.columns.iter().position(|c| c == "bucket").unwrap();

        // Dept B (Diana) should have bucket 1
        let dept_b: Vec<_> = result
            .rows
            .iter()
            .filter(|r| r.values[1] == Value::String("B".to_string()))
            .collect();
        assert_eq!(dept_b.len(), 1);
        assert_eq!(dept_b[0].values[bucket_idx], Value::Int(1));

        // Dept A has 3 rows, NTILE(2): buckets should be [1, 1, 2] or [1, 2, 2]
        let dept_a_buckets: Vec<i64> = result
            .rows
            .iter()
            .filter(|r| r.values[1] == Value::String("A".to_string()))
            .map(|r| {
                if let Value::Int(b) = &r.values[bucket_idx] {
                    *b
                } else {
                    0
                }
            })
            .collect();
        assert_eq!(dept_a_buckets.len(), 3);
        assert!(dept_a_buckets.contains(&1));
        assert!(dept_a_buckets.contains(&2));
    }

    #[test]
    fn test_window_first_value_with_partition() {
        let storage = setup_ranked_storage();
        let engine = QueryEngine::with_storage(storage);
        let ctx = QueryContext::new();

        let plan = ExecutionPlan::new(PlanNode::Window {
            input: Box::new(PlanNode::Scan {
                table: "scores".to_string(),
                columns: vec![],
                filter: None,
            }),
            window_functions: vec![crate::planner::WindowFunctionDef {
                alias: "top_in_dept".to_string(),
                function: "FIRST_VALUE".to_string(),
                args: vec![Expression::Column("name".to_string())],
                window: crate::ast::WindowSpec {
                    partition_by: vec![Expression::Column("dept".to_string())],
                    order_by: vec![crate::ast::OrderBy {
                        expr: Expression::Column("score".to_string()),
                        descending: true,
                        nulls_first: None,
                    }],
                    frame: None,
                },
            }],
        });

        let result = engine.execute_plan(&plan, &ctx).unwrap();
        let top_idx = result
            .columns
            .iter()
            .position(|c| c == "top_in_dept")
            .unwrap();

        // Dept B: only Diana -> FIRST_VALUE = Diana
        let dept_b: Vec<_> = result
            .rows
            .iter()
            .filter(|r| r.values[1] == Value::String("B".to_string()))
            .collect();
        assert_eq!(
            dept_b[0].values[top_idx],
            Value::String("Diana".to_string())
        );

        // Dept A: Alice(90)/Bob(90)/Charlie(80), ordered desc -> first is Alice or Bob
        let dept_a_first = result
            .rows
            .iter()
            .filter(|r| r.values[1] == Value::String("A".to_string()))
            .map(|r| r.values[top_idx].clone())
            .collect::<Vec<_>>();
        // All dept A rows should have the same FIRST_VALUE
        assert!(dept_a_first.iter().all(|v| *v == dept_a_first[0]));
        assert!(
            dept_a_first[0] == Value::String("Alice".to_string())
                || dept_a_first[0] == Value::String("Bob".to_string())
        );
    }
}
