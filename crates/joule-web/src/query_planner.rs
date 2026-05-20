//! SQL-like query planner — logical plan nodes (scan, filter, project, join,
//! sort, limit), plan optimization (predicate pushdown, projection pruning),
//! cost estimation, plan tree display, and plan serialization.
//!
//! Replaces query-plan libraries with pure Rust.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by the query planner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// A referenced column does not exist.
    UnknownColumn(String),
    /// Invalid plan structure.
    InvalidPlan(String),
    /// Cannot optimize: reason.
    OptimizationFailed(String),
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownColumn(c) => write!(f, "unknown column: {c}"),
            Self::InvalidPlan(msg) => write!(f, "invalid plan: {msg}"),
            Self::OptimizationFailed(msg) => write!(f, "optimization failed: {msg}"),
        }
    }
}

impl std::error::Error for PlanError {}

// ── Value type ───────────────────────────────────────────────────

/// Lightweight value for predicates.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "NULL"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Text(v) => write!(f, "'{v}'"),
            Self::Bool(v) => write!(f, "{v}"),
        }
    }
}

// ── Comparison operators ─────────────────────────────────────────

/// Comparison operator used in predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Eq => "=",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        };
        write!(f, "{s}")
    }
}

// ── Predicate ────────────────────────────────────────────────────

/// A filter predicate.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    /// column <op> literal
    Compare {
        column: String,
        op: CmpOp,
        value: Value,
    },
    /// AND of two predicates
    And(Box<Predicate>, Box<Predicate>),
    /// OR of two predicates
    Or(Box<Predicate>, Box<Predicate>),
    /// NOT predicate
    Not(Box<Predicate>),
    /// Always true
    True,
}

impl Predicate {
    /// Return all column names referenced by this predicate.
    pub fn referenced_columns(&self) -> HashSet<String> {
        let mut cols = HashSet::new();
        self.collect_columns(&mut cols);
        cols
    }

    fn collect_columns(&self, out: &mut HashSet<String>) {
        match self {
            Self::Compare { column, .. } => {
                out.insert(column.clone());
            }
            Self::And(a, b) | Self::Or(a, b) => {
                a.collect_columns(out);
                b.collect_columns(out);
            }
            Self::Not(inner) => inner.collect_columns(out),
            Self::True => {}
        }
    }

    /// Evaluate the predicate against a row represented as column-name -> Value map.
    pub fn evaluate(&self, row: &HashMap<String, Value>) -> bool {
        match self {
            Self::Compare { column, op, value } => {
                let cell = match row.get(column) {
                    Some(v) => v,
                    None => return false,
                };
                cmp_values(cell, value, *op)
            }
            Self::And(a, b) => a.evaluate(row) && b.evaluate(row),
            Self::Or(a, b) => a.evaluate(row) || b.evaluate(row),
            Self::Not(inner) => !inner.evaluate(row),
            Self::True => true,
        }
    }
}

fn cmp_values(left: &Value, right: &Value, op: CmpOp) -> bool {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => cmp_ord(a, b, op),
        (Value::Float(a), Value::Float(b)) => cmp_f64(*a, *b, op),
        (Value::Int(a), Value::Float(b)) => cmp_f64(*a as f64, *b, op),
        (Value::Float(a), Value::Int(b)) => cmp_f64(*a, *b as f64, op),
        (Value::Text(a), Value::Text(b)) => cmp_ord(a, b, op),
        (Value::Bool(a), Value::Bool(b)) => cmp_ord(a, b, op),
        (Value::Null, Value::Null) => matches!(op, CmpOp::Eq),
        _ => false,
    }
}

fn cmp_ord<T: Ord>(a: &T, b: &T, op: CmpOp) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }
}

fn cmp_f64(a: f64, b: f64, op: CmpOp) -> bool {
    match op {
        CmpOp::Eq => (a - b).abs() < f64::EPSILON,
        CmpOp::Ne => (a - b).abs() >= f64::EPSILON,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compare { column, op, value } => write!(f, "{column} {op} {value}"),
            Self::And(a, b) => write!(f, "({a} AND {b})"),
            Self::Or(a, b) => write!(f, "({a} OR {b})"),
            Self::Not(inner) => write!(f, "NOT ({inner})"),
            Self::True => write!(f, "TRUE"),
        }
    }
}

// ── Sort direction ───────────────────────────────────────────────

/// Sort order for ORDER BY.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDir {
    Asc,
    Desc,
}

impl fmt::Display for SortDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asc => write!(f, "ASC"),
            Self::Desc => write!(f, "DESC"),
        }
    }
}

/// A sort key: column + direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortKey {
    pub column: String,
    pub direction: SortDir,
}

// ── Join type ────────────────────────────────────────────────────

/// Type of join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

impl fmt::Display for JoinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inner => write!(f, "INNER"),
            Self::Left => write!(f, "LEFT"),
            Self::Right => write!(f, "RIGHT"),
            Self::Full => write!(f, "FULL"),
            Self::Cross => write!(f, "CROSS"),
        }
    }
}

// ── Logical plan ─────────────────────────────────────────────────

/// A logical plan node in the query plan tree.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// Table scan with table name and available columns.
    Scan {
        table: String,
        columns: Vec<String>,
    },
    /// Filter (WHERE).
    Filter {
        predicate: Predicate,
        input: Box<LogicalPlan>,
    },
    /// Projection (SELECT columns).
    Project {
        columns: Vec<String>,
        input: Box<LogicalPlan>,
    },
    /// Join two inputs.
    Join {
        join_type: JoinType,
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        left_col: String,
        right_col: String,
    },
    /// Sort (ORDER BY).
    Sort {
        keys: Vec<SortKey>,
        input: Box<LogicalPlan>,
    },
    /// Limit (LIMIT n [OFFSET m]).
    Limit {
        count: usize,
        offset: usize,
        input: Box<LogicalPlan>,
    },
}

impl LogicalPlan {
    /// Return the output columns of this plan node.
    pub fn output_columns(&self) -> Vec<String> {
        match self {
            Self::Scan { columns, .. } => columns.clone(),
            Self::Filter { input, .. } => input.output_columns(),
            Self::Project { columns, .. } => columns.clone(),
            Self::Join {
                left,
                right,
                ..
            } => {
                let mut cols = left.output_columns();
                cols.extend(right.output_columns());
                cols
            }
            Self::Sort { input, .. } => input.output_columns(),
            Self::Limit { input, .. } => input.output_columns(),
        }
    }

    /// Count the total number of nodes in the plan tree.
    pub fn node_count(&self) -> usize {
        match self {
            Self::Scan { .. } => 1,
            Self::Filter { input, .. }
            | Self::Project { input, .. }
            | Self::Sort { input, .. }
            | Self::Limit { input, .. } => 1 + input.node_count(),
            Self::Join { left, right, .. } => 1 + left.node_count() + right.node_count(),
        }
    }

    /// Return the depth of the plan tree.
    pub fn depth(&self) -> usize {
        match self {
            Self::Scan { .. } => 1,
            Self::Filter { input, .. }
            | Self::Project { input, .. }
            | Self::Sort { input, .. }
            | Self::Limit { input, .. } => 1 + input.depth(),
            Self::Join { left, right, .. } => {
                1 + std::cmp::max(left.depth(), right.depth())
            }
        }
    }

    /// Display the plan tree as an indented string.
    pub fn display_tree(&self) -> String {
        let mut out = String::new();
        self.format_tree(&mut out, 0);
        out
    }

    fn format_tree(&self, out: &mut String, indent: usize) {
        let pad: String = "  ".repeat(indent);
        match self {
            Self::Scan { table, columns } => {
                out.push_str(&format!("{pad}Scan({table} [{columns}])\n", columns = columns.join(", ")));
            }
            Self::Filter { predicate, input } => {
                out.push_str(&format!("{pad}Filter({predicate})\n"));
                input.format_tree(out, indent + 1);
            }
            Self::Project { columns, input } => {
                out.push_str(&format!("{pad}Project([{columns}])\n", columns = columns.join(", ")));
                input.format_tree(out, indent + 1);
            }
            Self::Join {
                join_type,
                left,
                right,
                left_col,
                right_col,
            } => {
                out.push_str(&format!(
                    "{pad}{join_type} Join({left_col} = {right_col})\n"
                ));
                left.format_tree(out, indent + 1);
                right.format_tree(out, indent + 1);
            }
            Self::Sort { keys, input } => {
                let key_str: Vec<String> =
                    keys.iter().map(|k| format!("{} {}", k.column, k.direction)).collect();
                out.push_str(&format!("{pad}Sort([{keys}])\n", keys = key_str.join(", ")));
                input.format_tree(out, indent + 1);
            }
            Self::Limit { count, offset, input } => {
                out.push_str(&format!("{pad}Limit({count}, offset={offset})\n"));
                input.format_tree(out, indent + 1);
            }
        }
    }

    /// Serialize the plan to a JSON-like map (serde_json::Value).
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Scan { table, columns } => serde_json::json!({
                "node": "Scan",
                "table": table,
                "columns": columns,
            }),
            Self::Filter { predicate, input } => serde_json::json!({
                "node": "Filter",
                "predicate": predicate.to_string(),
                "input": input.to_json(),
            }),
            Self::Project { columns, input } => serde_json::json!({
                "node": "Project",
                "columns": columns,
                "input": input.to_json(),
            }),
            Self::Join {
                join_type,
                left,
                right,
                left_col,
                right_col,
            } => serde_json::json!({
                "node": "Join",
                "join_type": join_type.to_string(),
                "left_col": left_col,
                "right_col": right_col,
                "left": left.to_json(),
                "right": right.to_json(),
            }),
            Self::Sort { keys, input } => {
                let key_arr: Vec<serde_json::Value> = keys
                    .iter()
                    .map(|k| serde_json::json!({
                        "column": k.column,
                        "direction": k.direction.to_string(),
                    }))
                    .collect();
                serde_json::json!({
                    "node": "Sort",
                    "keys": key_arr,
                    "input": input.to_json(),
                })
            }
            Self::Limit { count, offset, input } => serde_json::json!({
                "node": "Limit",
                "count": count,
                "offset": offset,
                "input": input.to_json(),
            }),
        }
    }
}

// ── Cost model ───────────────────────────────────────────────────

/// Table statistics used by the cost model.
#[derive(Debug, Clone)]
pub struct TableStats {
    pub table: String,
    pub row_count: usize,
    pub avg_row_bytes: usize,
    /// Distinct counts per column (if known).
    pub distinct: HashMap<String, usize>,
}

impl TableStats {
    pub fn new(table: &str, row_count: usize, avg_row_bytes: usize) -> Self {
        Self {
            table: table.to_string(),
            row_count,
            avg_row_bytes,
            distinct: HashMap::new(),
        }
    }

    pub fn with_distinct(mut self, column: &str, count: usize) -> Self {
        self.distinct.insert(column.to_string(), count);
        self
    }
}

/// Cost estimate for a plan.
#[derive(Debug, Clone, PartialEq)]
pub struct CostEstimate {
    pub estimated_rows: f64,
    pub estimated_cost: f64,
    pub estimated_bytes: f64,
}

/// Estimate the cost of a logical plan given table statistics.
pub fn estimate_cost(
    plan: &LogicalPlan,
    stats: &HashMap<String, TableStats>,
) -> CostEstimate {
    match plan {
        LogicalPlan::Scan { table, columns } => {
            let ts = stats.get(table);
            let rows = ts.map_or(1000.0, |s| s.row_count as f64);
            let avg_bytes = ts.map_or(100.0, |s| s.avg_row_bytes as f64);
            let col_fraction = ts.map_or(1.0, |s| {
                if s.distinct.is_empty() {
                    1.0
                } else {
                    columns.len() as f64 / s.distinct.len().max(columns.len()) as f64
                }
            });
            CostEstimate {
                estimated_rows: rows,
                estimated_cost: rows * col_fraction,
                estimated_bytes: rows * avg_bytes * col_fraction,
            }
        }
        LogicalPlan::Filter { predicate, input } => {
            let child = estimate_cost(input, stats);
            let selectivity = estimate_selectivity(predicate, stats);
            CostEstimate {
                estimated_rows: child.estimated_rows * selectivity,
                estimated_cost: child.estimated_cost + child.estimated_rows * 0.1,
                estimated_bytes: child.estimated_bytes * selectivity,
            }
        }
        LogicalPlan::Project { columns, input } => {
            let child = estimate_cost(input, stats);
            let total_cols = input.output_columns().len().max(1);
            let fraction = columns.len() as f64 / total_cols as f64;
            CostEstimate {
                estimated_rows: child.estimated_rows,
                estimated_cost: child.estimated_cost + child.estimated_rows * 0.01,
                estimated_bytes: child.estimated_bytes * fraction,
            }
        }
        LogicalPlan::Join { left, right, .. } => {
            let lc = estimate_cost(left, stats);
            let rc = estimate_cost(right, stats);
            let rows = lc.estimated_rows * rc.estimated_rows * 0.1;
            CostEstimate {
                estimated_rows: rows,
                estimated_cost: lc.estimated_cost + rc.estimated_cost + rows,
                estimated_bytes: rows * 200.0,
            }
        }
        LogicalPlan::Sort { input, keys } => {
            let child = estimate_cost(input, stats);
            let n = child.estimated_rows.max(1.0);
            let sort_cost = n * n.log2().max(1.0) * keys.len() as f64;
            CostEstimate {
                estimated_rows: child.estimated_rows,
                estimated_cost: child.estimated_cost + sort_cost,
                estimated_bytes: child.estimated_bytes,
            }
        }
        LogicalPlan::Limit { count, input, .. } => {
            let child = estimate_cost(input, stats);
            let rows = (*count as f64).min(child.estimated_rows);
            let fraction = if child.estimated_rows > 0.0 {
                rows / child.estimated_rows
            } else {
                1.0
            };
            CostEstimate {
                estimated_rows: rows,
                estimated_cost: child.estimated_cost + 1.0,
                estimated_bytes: child.estimated_bytes * fraction,
            }
        }
    }
}

fn estimate_selectivity(pred: &Predicate, _stats: &HashMap<String, TableStats>) -> f64 {
    match pred {
        Predicate::Compare { op, .. } => match op {
            CmpOp::Eq => 0.1,
            CmpOp::Ne => 0.9,
            CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => 0.3,
        },
        Predicate::And(a, b) => {
            estimate_selectivity(a, _stats) * estimate_selectivity(b, _stats)
        }
        Predicate::Or(a, b) => {
            let sa = estimate_selectivity(a, _stats);
            let sb = estimate_selectivity(b, _stats);
            (sa + sb - sa * sb).min(1.0)
        }
        Predicate::Not(inner) => 1.0 - estimate_selectivity(inner, _stats),
        Predicate::True => 1.0,
    }
}

// ── Optimizer ────────────────────────────────────────────────────

/// Optimize a logical plan. Applies predicate pushdown and projection pruning.
pub fn optimize(plan: LogicalPlan) -> LogicalPlan {
    let plan = push_down_predicates(plan);
    prune_projections(plan)
}

/// Push filter predicates below projections and joins where possible.
fn push_down_predicates(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::Filter { predicate, input } => {
            let input = push_down_predicates(*input);
            match input {
                // Push filter below project if columns are available.
                LogicalPlan::Project {
                    columns,
                    input: proj_input,
                } => {
                    let needed = predicate.referenced_columns();
                    let proj_cols: HashSet<String> =
                        columns.iter().cloned().collect();
                    if needed.is_subset(&proj_cols) {
                        // Push filter below project.
                        LogicalPlan::Project {
                            columns,
                            input: Box::new(LogicalPlan::Filter {
                                predicate,
                                input: proj_input,
                            }),
                        }
                    } else {
                        LogicalPlan::Filter {
                            predicate,
                            input: Box::new(LogicalPlan::Project {
                                columns,
                                input: proj_input,
                            }),
                        }
                    }
                }
                // Push filter below sort.
                LogicalPlan::Sort { keys, input: sort_input } => {
                    LogicalPlan::Sort {
                        keys,
                        input: Box::new(LogicalPlan::Filter {
                            predicate,
                            input: sort_input,
                        }),
                    }
                }
                other => LogicalPlan::Filter {
                    predicate,
                    input: Box::new(other),
                },
            }
        }
        LogicalPlan::Project { columns, input } => LogicalPlan::Project {
            columns,
            input: Box::new(push_down_predicates(*input)),
        },
        LogicalPlan::Sort { keys, input } => LogicalPlan::Sort {
            keys,
            input: Box::new(push_down_predicates(*input)),
        },
        LogicalPlan::Limit { count, offset, input } => LogicalPlan::Limit {
            count,
            offset,
            input: Box::new(push_down_predicates(*input)),
        },
        LogicalPlan::Join {
            join_type,
            left,
            right,
            left_col,
            right_col,
        } => LogicalPlan::Join {
            join_type,
            left: Box::new(push_down_predicates(*left)),
            right: Box::new(push_down_predicates(*right)),
            left_col,
            right_col,
        },
        other => other,
    }
}

/// Remove unused columns from projections when a parent projection is narrower.
fn prune_projections(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::Project { columns, input } => {
            let input = prune_projections(*input);
            // If the child is also a projection, merge.
            if let LogicalPlan::Project {
                columns: _child_cols,
                input: child_input,
            } = input
            {
                LogicalPlan::Project {
                    columns,
                    input: child_input,
                }
            } else {
                LogicalPlan::Project {
                    columns,
                    input: Box::new(input),
                }
            }
        }
        LogicalPlan::Filter { predicate, input } => LogicalPlan::Filter {
            predicate,
            input: Box::new(prune_projections(*input)),
        },
        LogicalPlan::Sort { keys, input } => LogicalPlan::Sort {
            keys,
            input: Box::new(prune_projections(*input)),
        },
        LogicalPlan::Limit { count, offset, input } => LogicalPlan::Limit {
            count,
            offset,
            input: Box::new(prune_projections(*input)),
        },
        LogicalPlan::Join {
            join_type,
            left,
            right,
            left_col,
            right_col,
        } => LogicalPlan::Join {
            join_type,
            left: Box::new(prune_projections(*left)),
            right: Box::new(prune_projections(*right)),
            left_col,
            right_col,
        },
        other => other,
    }
}

// ── Builder ──────────────────────────────────────────────────────

/// Fluent builder for constructing logical plans.
pub struct PlanBuilder {
    plan: LogicalPlan,
}

impl PlanBuilder {
    /// Start with a table scan.
    pub fn scan(table: &str, columns: &[&str]) -> Self {
        Self {
            plan: LogicalPlan::Scan {
                table: table.to_string(),
                columns: columns.iter().map(|c| c.to_string()).collect(),
            },
        }
    }

    /// Add a filter.
    pub fn filter(self, predicate: Predicate) -> Self {
        Self {
            plan: LogicalPlan::Filter {
                predicate,
                input: Box::new(self.plan),
            },
        }
    }

    /// Add a projection.
    pub fn project(self, columns: &[&str]) -> Self {
        Self {
            plan: LogicalPlan::Project {
                columns: columns.iter().map(|c| c.to_string()).collect(),
                input: Box::new(self.plan),
            },
        }
    }

    /// Add a sort.
    pub fn sort(self, keys: Vec<SortKey>) -> Self {
        Self {
            plan: LogicalPlan::Sort {
                keys,
                input: Box::new(self.plan),
            },
        }
    }

    /// Add a limit.
    pub fn limit(self, count: usize) -> Self {
        Self {
            plan: LogicalPlan::Limit {
                count,
                offset: 0,
                input: Box::new(self.plan),
            },
        }
    }

    /// Add a limit with offset.
    pub fn limit_offset(self, count: usize, offset: usize) -> Self {
        Self {
            plan: LogicalPlan::Limit {
                count,
                offset,
                input: Box::new(self.plan),
            },
        }
    }

    /// Join with another plan.
    pub fn join(
        self,
        right: PlanBuilder,
        join_type: JoinType,
        left_col: &str,
        right_col: &str,
    ) -> Self {
        Self {
            plan: LogicalPlan::Join {
                join_type,
                left: Box::new(self.plan),
                right: Box::new(right.plan),
                left_col: left_col.to_string(),
                right_col: right_col.to_string(),
            },
        }
    }

    /// Build the plan.
    pub fn build(self) -> LogicalPlan {
        self.plan
    }

    /// Build and optimize.
    pub fn build_optimized(self) -> LogicalPlan {
        optimize(self.plan)
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn users_scan() -> LogicalPlan {
        LogicalPlan::Scan {
            table: "users".into(),
            columns: vec!["id".into(), "name".into(), "age".into(), "email".into()],
        }
    }

    #[test]
    fn scan_output_columns() {
        let plan = users_scan();
        assert_eq!(plan.output_columns(), vec!["id", "name", "age", "email"]);
    }

    #[test]
    fn filter_preserves_columns() {
        let plan = LogicalPlan::Filter {
            predicate: Predicate::Compare {
                column: "age".into(),
                op: CmpOp::Gt,
                value: Value::Int(18),
            },
            input: Box::new(users_scan()),
        };
        assert_eq!(plan.output_columns().len(), 4);
    }

    #[test]
    fn project_narrows_columns() {
        let plan = LogicalPlan::Project {
            columns: vec!["id".into(), "name".into()],
            input: Box::new(users_scan()),
        };
        assert_eq!(plan.output_columns(), vec!["id", "name"]);
    }

    #[test]
    fn node_count() {
        let plan = PlanBuilder::scan("users", &["id", "name"])
            .filter(Predicate::True)
            .project(&["id"])
            .build();
        assert_eq!(plan.node_count(), 3);
    }

    #[test]
    fn depth_single() {
        let plan = users_scan();
        assert_eq!(plan.depth(), 1);
    }

    #[test]
    fn depth_chain() {
        let plan = PlanBuilder::scan("users", &["id"])
            .filter(Predicate::True)
            .sort(vec![SortKey {
                column: "id".into(),
                direction: SortDir::Asc,
            }])
            .limit(10)
            .build();
        assert_eq!(plan.depth(), 4);
    }

    #[test]
    fn join_output_columns() {
        let left = PlanBuilder::scan("users", &["id", "name"]);
        let right = PlanBuilder::scan("orders", &["order_id", "user_id"]);
        let plan = left.join(right, JoinType::Inner, "id", "user_id").build();
        assert_eq!(
            plan.output_columns(),
            vec!["id", "name", "order_id", "user_id"]
        );
    }

    #[test]
    fn predicate_evaluate_eq() {
        let pred = Predicate::Compare {
            column: "age".into(),
            op: CmpOp::Eq,
            value: Value::Int(30),
        };
        let mut row = HashMap::new();
        row.insert("age".into(), Value::Int(30));
        assert!(pred.evaluate(&row));
        row.insert("age".into(), Value::Int(25));
        assert!(!pred.evaluate(&row));
    }

    #[test]
    fn predicate_and_or() {
        let a = Predicate::Compare {
            column: "x".into(),
            op: CmpOp::Gt,
            value: Value::Int(0),
        };
        let b = Predicate::Compare {
            column: "y".into(),
            op: CmpOp::Lt,
            value: Value::Int(10),
        };
        let and_pred = Predicate::And(Box::new(a.clone()), Box::new(b.clone()));
        let mut row = HashMap::new();
        row.insert("x".into(), Value::Int(5));
        row.insert("y".into(), Value::Int(3));
        assert!(and_pred.evaluate(&row));

        let or_pred = Predicate::Or(Box::new(a), Box::new(b));
        row.insert("x".into(), Value::Int(-1));
        row.insert("y".into(), Value::Int(3));
        assert!(or_pred.evaluate(&row));
    }

    #[test]
    fn predicate_not() {
        let pred = Predicate::Not(Box::new(Predicate::True));
        let row = HashMap::new();
        assert!(!pred.evaluate(&row));
    }

    #[test]
    fn predicate_referenced_columns() {
        let pred = Predicate::And(
            Box::new(Predicate::Compare {
                column: "a".into(),
                op: CmpOp::Eq,
                value: Value::Int(1),
            }),
            Box::new(Predicate::Compare {
                column: "b".into(),
                op: CmpOp::Eq,
                value: Value::Int(2),
            }),
        );
        let cols = pred.referenced_columns();
        assert!(cols.contains("a"));
        assert!(cols.contains("b"));
        assert_eq!(cols.len(), 2);
    }

    #[test]
    fn cost_estimation_scan() {
        let plan = users_scan();
        let mut stats = HashMap::new();
        stats.insert(
            "users".into(),
            TableStats::new("users", 1000, 64),
        );
        let cost = estimate_cost(&plan, &stats);
        assert!((cost.estimated_rows - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cost_estimation_filter_reduces_rows() {
        let plan = LogicalPlan::Filter {
            predicate: Predicate::Compare {
                column: "age".into(),
                op: CmpOp::Eq,
                value: Value::Int(30),
            },
            input: Box::new(users_scan()),
        };
        let mut stats = HashMap::new();
        stats.insert("users".into(), TableStats::new("users", 1000, 64));
        let cost = estimate_cost(&plan, &stats);
        assert!(cost.estimated_rows < 1000.0);
    }

    #[test]
    fn cost_estimation_limit() {
        let plan = LogicalPlan::Limit {
            count: 10,
            offset: 0,
            input: Box::new(users_scan()),
        };
        let mut stats = HashMap::new();
        stats.insert("users".into(), TableStats::new("users", 1000, 64));
        let cost = estimate_cost(&plan, &stats);
        assert!((cost.estimated_rows - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn predicate_pushdown_below_project() {
        let plan = PlanBuilder::scan("users", &["id", "name", "age"])
            .project(&["id", "age"])
            .filter(Predicate::Compare {
                column: "age".into(),
                op: CmpOp::Gt,
                value: Value::Int(18),
            })
            .build();
        let optimized = optimize(plan);
        // After pushdown, the outer node should be Project, not Filter.
        match &optimized {
            LogicalPlan::Project { input, .. } => {
                assert!(matches!(input.as_ref(), LogicalPlan::Filter { .. }));
            }
            other => panic!("expected Project at top, got {:?}", other),
        }
    }

    #[test]
    fn projection_pruning_merges_nested() {
        let plan = PlanBuilder::scan("users", &["id", "name", "age"])
            .project(&["id", "name", "age"])
            .project(&["id"])
            .build();
        let optimized = optimize(plan);
        // Merged projections: top project directly on scan.
        match &optimized {
            LogicalPlan::Project { columns, input } => {
                assert_eq!(columns, &["id"]);
                assert!(matches!(input.as_ref(), LogicalPlan::Scan { .. }));
            }
            other => panic!("expected Project, got {:?}", other),
        }
    }

    #[test]
    fn display_tree_contains_nodes() {
        let plan = PlanBuilder::scan("users", &["id", "name"])
            .filter(Predicate::True)
            .limit(10)
            .build();
        let tree = plan.display_tree();
        assert!(tree.contains("Limit"));
        assert!(tree.contains("Filter"));
        assert!(tree.contains("Scan"));
    }

    #[test]
    fn to_json_roundtrip() {
        let plan = PlanBuilder::scan("users", &["id"]).limit(5).build();
        let json = plan.to_json();
        assert_eq!(json["node"], "Limit");
        assert_eq!(json["count"], 5);
        assert_eq!(json["input"]["node"], "Scan");
    }

    #[test]
    fn builder_full_chain() {
        let plan = PlanBuilder::scan("t", &["a", "b", "c"])
            .filter(Predicate::Compare {
                column: "a".into(),
                op: CmpOp::Eq,
                value: Value::Int(1),
            })
            .project(&["a", "b"])
            .sort(vec![SortKey {
                column: "a".into(),
                direction: SortDir::Asc,
            }])
            .limit_offset(10, 5)
            .build();
        assert_eq!(plan.node_count(), 5);
        assert_eq!(plan.output_columns(), vec!["a", "b"]);
    }

    #[test]
    fn predicate_display() {
        let pred = Predicate::Compare {
            column: "x".into(),
            op: CmpOp::Ge,
            value: Value::Float(3.14),
        };
        let s = pred.to_string();
        assert!(s.contains("x"));
        assert!(s.contains(">="));
        assert!(s.contains("3.14"));
    }

    #[test]
    fn cost_sort_increases_cost() {
        let scan = users_scan();
        let sort = LogicalPlan::Sort {
            keys: vec![SortKey {
                column: "name".into(),
                direction: SortDir::Asc,
            }],
            input: Box::new(users_scan()),
        };
        let mut stats = HashMap::new();
        stats.insert("users".into(), TableStats::new("users", 1000, 64));
        let scan_cost = estimate_cost(&scan, &stats);
        let sort_cost = estimate_cost(&sort, &stats);
        assert!(sort_cost.estimated_cost > scan_cost.estimated_cost);
    }

    #[test]
    fn filter_pushdown_below_sort() {
        let plan = PlanBuilder::scan("t", &["a"])
            .sort(vec![SortKey {
                column: "a".into(),
                direction: SortDir::Asc,
            }])
            .filter(Predicate::Compare {
                column: "a".into(),
                op: CmpOp::Eq,
                value: Value::Int(1),
            })
            .build();
        let optimized = optimize(plan);
        // Filter should be pushed below Sort.
        match &optimized {
            LogicalPlan::Sort { input, .. } => {
                assert!(matches!(input.as_ref(), LogicalPlan::Filter { .. }));
            }
            other => panic!("expected Sort at top, got {:?}", other),
        }
    }

    #[test]
    fn predicate_string_compare() {
        let pred = Predicate::Compare {
            column: "name".into(),
            op: CmpOp::Eq,
            value: Value::Text("Alice".into()),
        };
        let mut row = HashMap::new();
        row.insert("name".into(), Value::Text("Alice".into()));
        assert!(pred.evaluate(&row));
        row.insert("name".into(), Value::Text("Bob".into()));
        assert!(!pred.evaluate(&row));
    }
}
