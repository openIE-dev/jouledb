//! Query Planner

use crate::ast::{Expression, JoinType, OrderBy, Query, QueryType, WindowSpec};
use crate::error::{QueryError, QueryResult};
use crate::execution::ExecutionPlan;
use crate::sql::SetOperationType;
use serde::{Deserialize, Serialize};

/// Definition of a window function in the plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowFunctionDef {
    /// Output column alias
    pub alias: String,
    /// Function name (e.g., "ROW_NUMBER", "SUM", "AVG")
    pub function: String,
    /// Function arguments (expressions)
    pub args: Vec<Expression>,
    /// Window specification
    pub window: WindowSpec,
}

/// Distance metric for vector similarity search
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VectorMetric {
    L2,
    Cosine,
    InnerProduct,
}

/// Cascade tier classification for a query plan.
///
/// Maps SQL `PlanNode` shapes to the energy-aware cascade tiers documented in
/// `docs/MGAI-SPEC-DOMAIN-JOULEDB.md` and the v0.2 whitepaper. The ordering is
/// energy-ascending: `Lookup` is cheapest (µJ-class), `Reason` is most expensive
/// (J-class). The top-level cascade tier for a query is the max over its plan
/// tree — i.e. the most expensive operation determines the query's tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CascadeTier {
    /// Single-row index lookup, primary-key fetch, cache hit. ~0.1 µJ.
    Lookup = 0,
    /// Closed-form computation, no I/O (constant arithmetic, scalar expr). ~1 µJ.
    Formula = 1,
    /// Table or index scan, range read. ~µJ-mJ.
    Extract = 2,
    /// Joins, aggregations, sorts, window functions, set ops. ~mJ-J.
    Aggregate = 3,
    /// Recursive CTEs and multi-stage iteration. ~J.
    Reason = 4,
}

impl CascadeTier {
    /// Human-readable name (matches `InferenceHorizon` casing in `ask-core`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Lookup => "Lookup",
            Self::Formula => "Formula",
            Self::Extract => "Extract",
            Self::Aggregate => "Aggregate",
            Self::Reason => "Reason",
        }
    }
}

impl std::fmt::Display for CascadeTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Plan node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanNode {
    /// Table scan
    Scan {
        table: String,
        columns: Vec<String>,
        filter: Option<Expression>,
    },
    /// Index scan
    IndexScan {
        table: String,
        index: String,
        columns: Vec<String>,
        filter: Expression,
    },
    /// Filter operation
    Filter {
        input: Box<PlanNode>,
        predicate: Expression,
    },
    /// Projection (column selection)
    Project {
        input: Box<PlanNode>,
        columns: Vec<String>,
    },
    /// Sort operation
    Sort {
        input: Box<PlanNode>,
        order_by: Vec<OrderBy>,
    },
    /// Limit operation
    Limit {
        input: Box<PlanNode>,
        limit: Option<usize>,
        offset: Option<usize>,
    },
    /// Aggregation
    Aggregate {
        input: Box<PlanNode>,
        group_by: Vec<Expression>,
        aggregates: Vec<(String, String, String)>, // (alias, function, column)
    },
    /// Join operation
    Join {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_type: JoinType,
        condition: Option<Expression>,
    },
    /// Window function operation
    Window {
        input: Box<PlanNode>,
        /// List of window functions: (output_alias, function_name, args, window_spec)
        window_functions: Vec<WindowFunctionDef>,
    },
    /// Set operation (UNION, EXCEPT, INTERSECT) of two result sets
    SetOp {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        op: SetOperationType,
    },
    /// Distinct (deduplicate rows)
    Distinct { input: Box<PlanNode> },
    /// Recursive CTE: iteratively execute recursive part until fixed point
    RecursiveCte {
        /// CTE name (used for self-reference resolution)
        name: String,
        /// Base (anchor) plan
        base: Box<PlanNode>,
        /// Recursive plan (references the CTE name)
        recursive: Box<PlanNode>,
        /// Column names for the CTE
        columns: Vec<String>,
    },
    /// Vector similarity search (KNN via HNSW/IVF index)
    VectorScan {
        table: String,
        vector_column: String,
        query_vector: Vec<f32>,
        metric: VectorMetric,
        k: usize,
        filter: Option<Expression>,
    },
    /// Worst-case optimal join for cyclic graph patterns (triangles, cliques).
    /// Uses Leapfrog TrieJoin for multi-way intersection.
    WcojJoin {
        /// Atoms: (relation_name, variable_bindings)
        atoms: Vec<(String, Vec<String>)>,
        /// Output variable names
        output_variables: Vec<String>,
    },
    /// Empty result
    Empty,
}

impl PlanNode {
    /// Create scan node
    pub fn scan(table: &str) -> Self {
        Self::Scan {
            table: table.to_string(),
            columns: Vec::new(),
            filter: None,
        }
    }

    /// Add filter to node
    pub fn filter(self, predicate: Expression) -> Self {
        Self::Filter {
            input: Box::new(self),
            predicate,
        }
    }

    /// Add projection to node
    pub fn project(self, columns: Vec<String>) -> Self {
        Self::Project {
            input: Box::new(self),
            columns,
        }
    }

    /// Add sort to node
    pub fn sort(self, order_by: Vec<OrderBy>) -> Self {
        Self::Sort {
            input: Box::new(self),
            order_by,
        }
    }

    /// Add limit to node
    pub fn limit(self, limit: Option<usize>, offset: Option<usize>) -> Self {
        Self::Limit {
            input: Box::new(self),
            limit,
            offset,
        }
    }

    /// Estimate cost of this node using statistics
    pub fn estimate_cost(&self) -> f64 {
        self.estimate_cost_with_stats(&Statistics::default())
    }

    /// Estimate cost with statistics catalog
    pub fn estimate_cost_with_stats(&self, stats: &Statistics) -> f64 {
        match self {
            Self::Scan { table, filter, .. } => {
                let base_cost = stats.table_rows(table) as f64 * COST_SEQ_PAGE_ACCESS;
                if filter.is_some() {
                    base_cost * stats.selectivity(filter.as_ref())
                } else {
                    base_cost
                }
            }
            Self::IndexScan { table, filter, .. } => {
                // Index access cost: traverse tree + read matching pages
                // Very simplified: Log(N) + (Selectivity * N)
                let total_rows = stats.table_rows(table) as f64;
                // Avoid log2(0)
                let tree_height = if total_rows > 1.0 {
                    total_rows.log2()
                } else {
                    1.0
                };
                let selectivity = stats.selectivity(Some(filter));
                let matching_rows = total_rows * selectivity;

                (tree_height * COST_RANDOM_PAGE_ACCESS)
                    + (matching_rows * COST_SEQ_PAGE_ACCESS * 0.1)
            }
            Self::Filter {
                input,
                predicate: _,
            } => {
                let input_cost = input.estimate_cost_with_stats(stats);
                let input_rows = input.estimate_rows_with_stats(stats) as f64;
                input_cost + input_rows * COST_CPU_TUPLE
            }
            Self::Project { input, columns } => {
                let input_cost = input.estimate_cost_with_stats(stats);
                let input_rows = input.estimate_rows_with_stats(stats) as f64;
                input_cost + input_rows * COST_CPU_TUPLE * columns.len() as f64 * 0.01
            }
            Self::Sort { input, order_by } => {
                let input_cost = input.estimate_cost_with_stats(stats);
                let input_rows = input.estimate_rows_with_stats(stats) as f64;
                // O(n log n) sort cost
                let sort_cost = if input_rows > 1.0 {
                    input_rows * input_rows.log2() * COST_CPU_TUPLE * order_by.len() as f64
                } else {
                    0.0
                };
                input_cost + sort_cost
            }
            Self::Limit { input, .. } => input.estimate_cost_with_stats(stats) + COST_CPU_TUPLE,
            Self::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                let input_cost = input.estimate_cost_with_stats(stats);
                let input_rows = input.estimate_rows_with_stats(stats) as f64;
                let agg_cost =
                    input_rows * (group_by.len() + aggregates.len()) as f64 * COST_CPU_TUPLE;
                input_cost + agg_cost
            }
            Self::Join {
                left,
                right,
                join_type,
                ..
            } => {
                let left_cost = left.estimate_cost_with_stats(stats);
                let right_cost = right.estimate_cost_with_stats(stats);
                let left_rows = left.estimate_rows_with_stats(stats) as f64;
                let right_rows = right.estimate_rows_with_stats(stats) as f64;

                // Nested loop join cost estimation
                let join_cost = match join_type {
                    JoinType::Inner | JoinType::Left | JoinType::Right => {
                        // Hash join cost for larger joins, nested loop for small
                        if left_rows * right_rows > 10000.0 {
                            // Hash join: build hash table + probe
                            left_rows * COST_HASH_BUILD + right_rows * COST_HASH_PROBE
                        } else {
                            // Nested loop
                            left_rows * right_rows * COST_CPU_TUPLE
                        }
                    }
                    JoinType::Full => left_rows * right_rows * COST_CPU_TUPLE * 2.0,
                    JoinType::Cross => left_rows * right_rows * COST_CPU_TUPLE,
                };
                left_cost + right_cost + join_cost
            }
            Self::Window {
                input,
                window_functions,
            } => {
                let input_cost = input.estimate_cost_with_stats(stats);
                let input_rows = input.estimate_rows_with_stats(stats) as f64;
                // Window functions require sorting and scanning
                let window_cost =
                    input_rows * window_functions.len() as f64 * COST_CPU_TUPLE * 10.0;
                input_cost + window_cost
            }
            Self::SetOp { left, right, op } => {
                let left_cost = left.estimate_cost_with_stats(stats);
                let right_cost = right.estimate_cost_with_stats(stats);
                let needs_dedup = !matches!(
                    op,
                    SetOperationType::UnionAll
                        | SetOperationType::ExceptAll
                        | SetOperationType::IntersectAll
                );
                let dedup_cost = if needs_dedup {
                    let total_rows = left.estimate_rows_with_stats(stats)
                        + right.estimate_rows_with_stats(stats);
                    total_rows as f64 * COST_CPU_TUPLE * 2.0
                } else {
                    0.0
                };
                left_cost + right_cost + dedup_cost
            }
            Self::Distinct { input } => {
                let input_cost = input.estimate_cost_with_stats(stats);
                let input_rows = input.estimate_rows_with_stats(stats) as f64;
                input_cost + input_rows * COST_CPU_TUPLE * 2.0
            }
            Self::RecursiveCte {
                base, recursive, ..
            } => {
                let base_cost = base.estimate_cost_with_stats(stats);
                let recursive_cost = recursive.estimate_cost_with_stats(stats);
                base_cost + recursive_cost * 10.0 // assume ~10 iterations
            }
            Self::VectorScan { k, .. } => {
                // Approximate: log2(N) * k for HNSW-style search
                let n = stats.table_rows(&String::new()) as f64;
                if n > 0.0 {
                    n.log2() * (*k as f64)
                } else {
                    *k as f64
                }
            }
            Self::WcojJoin { atoms, .. } => {
                // WCOJ cost: proportional to output size, bounded by AGM bound.
                // Heuristic: atoms.len() * base_cost
                atoms.len() as f64 * 100.0
            }
            Self::Empty => 0.0,
        }
    }

    /// Estimate row count
    pub fn estimate_rows(&self) -> usize {
        self.estimate_rows_with_stats(&Statistics::default())
    }

    /// Estimate row count with statistics
    pub fn estimate_rows_with_stats(&self, stats: &Statistics) -> usize {
        match self {
            Self::Scan { table, filter, .. } => {
                let base_rows = stats.table_rows(table);
                if let Some(f) = filter {
                    (base_rows as f64 * stats.selectivity(Some(f))) as usize
                } else {
                    base_rows
                }
            }
            Self::IndexScan { table, filter, .. } => {
                let base_rows = stats.table_rows(table);
                (base_rows as f64 * stats.selectivity(Some(filter))) as usize
            }
            Self::Filter { input, predicate } => {
                let input_rows = input.estimate_rows_with_stats(stats) as f64;
                (input_rows * stats.selectivity(Some(predicate))) as usize
            }
            Self::Project { input, .. } => input.estimate_rows_with_stats(stats),
            Self::Sort { input, .. } => input.estimate_rows_with_stats(stats),
            Self::Limit {
                input,
                limit,
                offset,
            } => {
                let rows = input.estimate_rows_with_stats(stats);
                let start = offset.unwrap_or(0);
                let end = limit.map(|l| start + l).unwrap_or(rows);
                end.saturating_sub(start).min(rows)
            }
            Self::Aggregate {
                input, group_by, ..
            } => {
                if group_by.is_empty() {
                    1
                } else {
                    let input_rows = input.estimate_rows_with_stats(stats);
                    // Estimate distinct groups — extract column name if possible
                    let col_name = match &group_by[0] {
                        Expression::Column(name) => Some(name.as_str()),
                        _ => None,
                    };
                    let ndv = col_name
                        .map(|n| stats.distinct_values(n))
                        .unwrap_or(input_rows);
                    ndv.min(input_rows)
                }
            }
            Self::Join {
                left,
                right,
                join_type,
                condition,
            } => {
                let left_rows = left.estimate_rows_with_stats(stats);
                let right_rows = right.estimate_rows_with_stats(stats);

                match join_type {
                    JoinType::Inner => {
                        // Estimate based on condition selectivity
                        let selectivity = stats.selectivity(condition.as_ref());
                        ((left_rows * right_rows) as f64 * selectivity) as usize
                    }
                    JoinType::Left => left_rows,
                    JoinType::Right => right_rows,
                    JoinType::Full => left_rows + right_rows,
                    JoinType::Cross => left_rows * right_rows,
                }
            }
            Self::Window { input, .. } => input.estimate_rows_with_stats(stats),
            Self::SetOp { left, right, op } => {
                let left_rows = left.estimate_rows_with_stats(stats);
                let right_rows = right.estimate_rows_with_stats(stats);
                match op {
                    SetOperationType::UnionAll => left_rows + right_rows,
                    SetOperationType::Union => ((left_rows + right_rows) as f64 * 0.8) as usize,
                    SetOperationType::Except | SetOperationType::ExceptAll => {
                        // Assume ~50% of left rows remain after removing right
                        (left_rows as f64 * 0.5).max(1.0) as usize
                    }
                    SetOperationType::Intersect | SetOperationType::IntersectAll => {
                        // Assume ~30% overlap
                        (left_rows.min(right_rows) as f64 * 0.3).max(1.0) as usize
                    }
                }
            }
            Self::Distinct { input } => {
                // Estimate ~80% of input rows are distinct
                let input_rows = input.estimate_rows_with_stats(stats);
                (input_rows as f64 * 0.8).max(1.0) as usize
            }
            Self::RecursiveCte { base, .. } => {
                // Estimate: base rows * ~10 iterations
                base.estimate_rows_with_stats(stats) * 10
            }
            Self::VectorScan { k, .. } => *k,
            Self::WcojJoin { atoms, .. } => {
                // Heuristic: product of atom counts, bounded
                atoms.len() * 100
            }
            Self::Empty => 0,
        }
    }

    /// Classify this plan node to its cascade tier — the energy-class
    /// of operation it represents. Recurses into children; the returned
    /// tier is the **max** across this node and its descendants, since
    /// the most expensive op determines the query's tier.
    ///
    /// Closes Open Item §10.5 in `docs/MGAI-SPEC-DOMAIN-JOULEDB.md`.
    pub fn cascade_tier(&self) -> CascadeTier {
        let own = self.own_cascade_tier();
        let children = self.children_max_cascade_tier();
        own.max(children)
    }

    /// This node's own contribution to the cascade tier, ignoring children.
    fn own_cascade_tier(&self) -> CascadeTier {
        match self {
            Self::Empty => CascadeTier::Formula,
            Self::Scan { filter, .. } => {
                // PK / unique-index-equality filter looks like a single-row
                // lookup; everything else is a range scan.
                if filter.as_ref().is_some_and(is_unique_lookup) {
                    CascadeTier::Lookup
                } else {
                    CascadeTier::Extract
                }
            }
            Self::IndexScan { filter, .. } => {
                if is_unique_lookup(filter) {
                    CascadeTier::Lookup
                } else {
                    CascadeTier::Extract
                }
            }
            Self::Filter { .. } | Self::Project { .. } | Self::Limit { .. } => {
                // Pass-through; transparent to the cascade tier. Contributes
                // the floor so the input's tier dominates entirely (a
                // projection over a PK lookup is still a Lookup).
                CascadeTier::Lookup
            }
            Self::Sort { .. }
            | Self::Aggregate { .. }
            | Self::Window { .. }
            | Self::Distinct { .. }
            | Self::SetOp { .. }
            | Self::WcojJoin { .. } => CascadeTier::Aggregate,
            Self::Join { .. } => CascadeTier::Aggregate,
            Self::VectorScan { .. } => CascadeTier::Extract,
            Self::RecursiveCte { .. } => CascadeTier::Reason,
        }
    }

    /// Max cascade tier across all direct children.
    fn children_max_cascade_tier(&self) -> CascadeTier {
        match self {
            Self::Scan { .. } | Self::IndexScan { .. } | Self::Empty | Self::VectorScan { .. } => {
                CascadeTier::Lookup
            }
            Self::Filter { input, .. }
            | Self::Project { input, .. }
            | Self::Sort { input, .. }
            | Self::Limit { input, .. }
            | Self::Aggregate { input, .. }
            | Self::Window { input, .. }
            | Self::Distinct { input } => input.cascade_tier(),
            Self::Join { left, right, .. } | Self::SetOp { left, right, .. } => {
                left.cascade_tier().max(right.cascade_tier())
            }
            Self::RecursiveCte { base, recursive, .. } => {
                base.cascade_tier().max(recursive.cascade_tier())
            }
            Self::WcojJoin { .. } => CascadeTier::Lookup,
        }
    }
}

/// Recognise a unique-row lookup pattern (`col = literal` on a PK / unique index).
fn is_unique_lookup(expr: &Expression) -> bool {
    use crate::ast::Operator;
    match expr {
        Expression::Binary { left, op: Operator::Eq, right } => {
            matches!(
                (left.as_ref(), right.as_ref()),
                (Expression::Column(_), Expression::Literal(_))
                    | (Expression::Literal(_), Expression::Column(_))
                    | (Expression::QualifiedColumn { .. }, Expression::Literal(_))
                    | (Expression::Literal(_), Expression::QualifiedColumn { .. })
            )
        }
        Expression::Binary { left, op: Operator::And, right } => {
            // Both sides equality on a column — likely composite PK lookup.
            is_unique_lookup(left) && is_unique_lookup(right)
        }
        _ => false,
    }
}

// Cost model constants
const COST_SEQ_PAGE_ACCESS: f64 = 1.0;
#[allow(dead_code)]
const COST_RANDOM_PAGE_ACCESS: f64 = 4.0;
const COST_CPU_TUPLE: f64 = 0.01;
const COST_HASH_BUILD: f64 = 0.05;
const COST_HASH_PROBE: f64 = 0.02;
#[allow(dead_code)]
const COST_INDEX_ACCESS: f64 = 0.5;

/// Table and column statistics for cost estimation
#[derive(Debug, Clone, Default)]
pub struct Statistics {
    /// Row counts per table
    table_stats: std::collections::HashMap<String, TableStats>,
}

/// Statistics for a single table
#[derive(Debug, Clone)]
pub struct TableStats {
    /// Total row count
    pub row_count: usize,
    /// Column statistics
    pub columns: std::collections::HashMap<String, ColumnStats>,
    /// Available indexes
    pub indexes: Vec<IndexInfo>,
}

impl Default for TableStats {
    fn default() -> Self {
        Self {
            row_count: 1000,
            columns: std::collections::HashMap::new(),
            indexes: Vec::new(),
        }
    }
}

/// Statistics for a single column
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Number of distinct values
    pub distinct_count: usize,
    /// Null fraction (0.0 to 1.0)
    pub null_fraction: f64,
    /// Minimum value (if applicable)
    pub min_value: Option<String>,
    /// Maximum value (if applicable)
    pub max_value: Option<String>,
    /// Most common values with frequencies
    pub most_common: Vec<(String, f64)>,
    /// Histogram bucket boundaries
    pub histogram: Vec<f64>,
}

impl Default for ColumnStats {
    fn default() -> Self {
        Self {
            distinct_count: 100,
            null_fraction: 0.0,
            min_value: None,
            max_value: None,
            most_common: Vec::new(),
            histogram: Vec::new(),
        }
    }
}

/// Index information
#[derive(Debug, Clone)]
pub struct IndexInfo {
    /// Index name
    pub name: String,
    /// Columns in the index
    pub columns: Vec<String>,
    /// Whether this is a unique index
    pub unique: bool,
    /// Index type
    pub index_type: IndexType,
}

/// Index type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    BTree,
    Hash,
    Bitmap,
    FullText,
    Spatial,
    Vector,
    /// GIN (Generalized Inverted Index) for JSONB containment, array overlap, full-text, trigram
    Gin,
    /// Adaptive Radix Tree — cache-friendly ordered index, 3-5x faster lookups than BTree
    ART,
    /// MinHash-LSH — near-duplicate detection and Jaccard similarity queries
    MinHashLSH,
}

impl Statistics {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Add table statistics
    pub fn add_table(&mut self, name: &str, stats: TableStats) {
        self.table_stats.insert(name.to_string(), stats);
    }

    /// Get row count for a table
    pub fn table_rows(&self, table: &str) -> usize {
        self.table_stats
            .get(table)
            .map(|s| s.row_count)
            .unwrap_or(1000)
    }

    /// Get distinct value count for a column
    pub fn distinct_values(&self, column: &str) -> usize {
        // Parse table.column format
        let parts: Vec<_> = column.split('.').collect();
        if parts.len() == 2 {
            if let Some(table) = self.table_stats.get(parts[0]) {
                if let Some(col) = table.columns.get(parts[1]) {
                    return col.distinct_count;
                }
            }
        }
        100 // Default NDV
    }

    /// Estimate selectivity of a predicate
    pub fn selectivity(&self, predicate: Option<&Expression>) -> f64 {
        match predicate {
            None => 1.0,
            Some(expr) => self.estimate_selectivity(expr),
        }
    }

    fn estimate_selectivity(&self, expr: &Expression) -> f64 {
        match expr {
            Expression::Binary { op, left, right } => {
                match op {
                    crate::ast::Operator::Eq => {
                        // Point query: 1/NDV
                        if let Expression::Column(col) = left.as_ref() {
                            let ndv = self.distinct_values(col) as f64;
                            return 1.0 / ndv.max(1.0);
                        }
                        0.1 // Default
                    }
                    crate::ast::Operator::Ne => {
                        1.0 - self.estimate_selectivity(&Expression::Binary {
                            op: crate::ast::Operator::Eq,
                            left: left.clone(),
                            right: right.clone(),
                        })
                    }
                    crate::ast::Operator::Lt
                    | crate::ast::Operator::Le
                    | crate::ast::Operator::Gt
                    | crate::ast::Operator::Ge => {
                        0.33 // Range predicate default
                    }
                    crate::ast::Operator::And => {
                        // Independence assumption
                        self.estimate_selectivity(left) * self.estimate_selectivity(right)
                    }
                    crate::ast::Operator::Or => {
                        // P(A or B) = P(A) + P(B) - P(A and B)
                        let sel_a = self.estimate_selectivity(left);
                        let sel_b = self.estimate_selectivity(right);
                        sel_a + sel_b - (sel_a * sel_b)
                    }
                    _ => 0.5,
                }
            }
            Expression::IsNull { negated, .. } => {
                if *negated {
                    0.95
                } else {
                    0.05
                }
            }
            Expression::In { list, negated, .. } => {
                let selectivity = (list.len() as f64 * 0.05).min(0.5);
                if *negated {
                    1.0 - selectivity
                } else {
                    selectivity
                }
            }
            Expression::Like { .. } => 0.1, // LIKE usually matches few rows
            Expression::Between { .. } => 0.25, // Range
            Expression::Unary { op, expr } => match op {
                crate::ast::UnaryOperator::Not => 1.0 - self.estimate_selectivity(expr),
                _ => self.estimate_selectivity(expr),
            },
            _ => 0.5, // Default selectivity
        }
    }

    /// Get available indexes for a table
    pub fn get_indexes(&self, table: &str) -> &[IndexInfo] {
        self.table_stats
            .get(table)
            .map(|s| s.indexes.as_slice())
            .unwrap_or(&[])
    }

    /// Check if a B-tree or Hash index can be used for a standard comparison predicate.
    ///
    /// Only matches B-tree/Hash indexes — vector, spatial, and full-text indexes
    /// require different scan patterns and are selected via dedicated methods.
    pub fn can_use_index(&self, table: &str, predicate: &Expression) -> Option<&IndexInfo> {
        let indexes = self.get_indexes(table);
        let column = self.extract_column_from_predicate(predicate)?;

        indexes.iter().find(|idx| {
            idx.columns.first() == Some(&column)
                && matches!(idx.index_type, IndexType::BTree | IndexType::Hash)
        })
    }

    /// Check if a vector index exists for the given table and column.
    pub fn can_use_vector_index(&self, table: &str, column: &str) -> Option<&IndexInfo> {
        let indexes = self.get_indexes(table);
        indexes.iter().find(|idx| {
            idx.columns.first().map(|s| s.as_str()) == Some(column)
                && idx.index_type == IndexType::Vector
        })
    }

    /// Check if a full-text index exists for the given table and column.
    pub fn can_use_fts_index(&self, table: &str, column: &str) -> Option<&IndexInfo> {
        let indexes = self.get_indexes(table);
        indexes.iter().find(|idx| {
            idx.columns.iter().any(|c| c == column) && idx.index_type == IndexType::FullText
        })
    }

    /// Check if a spatial index exists for the given table and column.
    pub fn can_use_spatial_index(&self, table: &str, column: &str) -> Option<&IndexInfo> {
        let indexes = self.get_indexes(table);
        indexes.iter().find(|idx| {
            idx.columns.first().map(|s| s.as_str()) == Some(column)
                && idx.index_type == IndexType::Spatial
        })
    }

    /// Check if a GIN index can accelerate the given predicate for a table.
    ///
    /// Matches predicates containing `@>` (JsonContains), `<@` (JsonContainedBy),
    /// or LIKE patterns (trigram acceleration).
    pub fn can_use_gin_index(&self, table: &str, predicate: &Expression) -> Option<&IndexInfo> {
        let column = self.extract_gin_column(predicate)?;
        let indexes = self.get_indexes(table);
        indexes.iter().find(|idx| {
            idx.columns.first().map(|s| s.as_str()) == Some(column.as_str())
                && idx.index_type == IndexType::Gin
        })
    }

    /// Extract the column name from a GIN-eligible predicate.
    fn extract_gin_column(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::Binary { op, left, right } => {
                match op {
                    crate::ast::Operator::JsonContains | crate::ast::Operator::JsonContainedBy => {
                        if let Expression::Column(col) = left.as_ref() {
                            return Some(col.clone());
                        }
                        None
                    }
                    crate::ast::Operator::And => {
                        // Recurse into AND — find any GIN-eligible branch
                        self.extract_gin_column(left)
                            .or_else(|| self.extract_gin_column(right))
                    }
                    _ => None,
                }
            }
            Expression::Like { expr: inner, .. } => {
                if let Expression::Column(col) = inner.as_ref() {
                    Some(col.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn extract_column_from_predicate(&self, expr: &Expression) -> Option<String> {
        match expr {
            Expression::Binary { left, .. } => {
                if let Expression::Column(col) = left.as_ref() {
                    return Some(col.clone());
                }
                None
            }
            _ => None,
        }
    }
}

/// Query planner with cost-based optimization
pub struct QueryPlanner {
    /// Enable cost-based optimization
    pub cost_based: bool,
    /// Enable predicate pushdown
    pub predicate_pushdown: bool,
    /// Enable projection pushdown
    pub projection_pushdown: bool,
    /// Enable join reordering
    pub join_reorder: bool,
    /// Enable index selection
    pub use_indexes: bool,
    /// Statistics catalog
    pub statistics: Statistics,
}

impl QueryPlanner {
    /// Create new planner
    pub fn new() -> Self {
        Self {
            cost_based: true,
            predicate_pushdown: true,
            projection_pushdown: true,
            join_reorder: true,
            use_indexes: true,
            statistics: Statistics::default(),
        }
    }

    /// Create planner with statistics
    pub fn with_statistics(stats: Statistics) -> Self {
        Self {
            statistics: stats,
            ..Self::new()
        }
    }

    /// Plan a query
    pub fn plan(&self, query: &Query) -> QueryResult<ExecutionPlan> {
        let root = self.build_plan(query, &[])?;
        let optimized = self.optimize(root)?;

        Ok(ExecutionPlan {
            estimated_cost: optimized.estimate_cost_with_stats(&self.statistics),
            estimated_rows: optimized.estimate_rows_with_stats(&self.statistics),
            root: optimized,
            warnings: Vec::new(),
        })
    }

    /// Explain the query plan
    pub fn explain(&self, query: &Query) -> QueryResult<String> {
        let plan = self.plan(query)?;
        Ok(self.format_plan(&plan.root, 0))
    }

    /// Format a plan node tree as a human-readable string with cost estimates.
    pub fn format_plan(&self, node: &PlanNode, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        let cost = node.estimate_cost_with_stats(&self.statistics);
        let rows = node.estimate_rows_with_stats(&self.statistics);

        match node {
            PlanNode::Scan {
                table,
                columns,
                filter,
            } => {
                let cols = if columns.is_empty() {
                    "*".to_string()
                } else {
                    columns.join(", ")
                };
                let filter_str = filter
                    .as_ref()
                    .map(|f| format!(" WHERE {:?}", f))
                    .unwrap_or_default();
                format!(
                    "{}Scan {} [{}]{} (cost={:.2}, rows={})",
                    prefix, table, cols, filter_str, cost, rows
                )
            }
            PlanNode::IndexScan {
                table,
                index,
                columns,
                filter,
            } => {
                let cols = if columns.is_empty() {
                    "*".to_string()
                } else {
                    columns.join(", ")
                };
                format!(
                    "{}IndexScan {} using {} [{}]{} (cost={:.2}, rows={})",
                    prefix,
                    table,
                    index,
                    cols,
                    format!(" WHERE {:?}", filter),
                    cost,
                    rows
                )
            }
            PlanNode::Filter { input, predicate } => {
                let child = self.format_plan(input, indent + 1);
                format!(
                    "{}Filter {:?} (cost={:.2}, rows={})\n{}",
                    prefix, predicate, cost, rows, child
                )
            }
            PlanNode::Project { input, columns } => {
                let child = self.format_plan(input, indent + 1);
                format!(
                    "{}Project [{}] (cost={:.2}, rows={})\n{}",
                    prefix,
                    columns.join(", "),
                    cost,
                    rows,
                    child
                )
            }
            PlanNode::Sort { input, order_by } => {
                let child = self.format_plan(input, indent + 1);
                let order: Vec<_> = order_by
                    .iter()
                    .map(|o| {
                        let col = match &o.expr {
                            Expression::Column(name) => name.clone(),
                            _ => format!("{:?}", o.expr),
                        };
                        format!("{} {}", col, if o.descending { "DESC" } else { "ASC" })
                    })
                    .collect();
                format!(
                    "{}Sort [{}] (cost={:.2}, rows={})\n{}",
                    prefix,
                    order.join(", "),
                    cost,
                    rows,
                    child
                )
            }
            PlanNode::Limit {
                input,
                limit,
                offset,
            } => {
                let child = self.format_plan(input, indent + 1);
                format!(
                    "{}Limit {} offset {} (cost={:.2}, rows={})\n{}",
                    prefix,
                    limit.unwrap_or(0),
                    offset.unwrap_or(0),
                    cost,
                    rows,
                    child
                )
            }
            PlanNode::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                let child = self.format_plan(input, indent + 1);
                let aggs: Vec<_> = aggregates
                    .iter()
                    .map(|(_a, f, _c)| format!("{}", f))
                    .collect();
                format!(
                    "{}Aggregate group=[{}] aggs=[{}] (cost={:.2}, rows={})\n{}",
                    prefix,
                    group_by
                        .iter()
                        .map(|e| format!("{:?}", e))
                        .collect::<Vec<_>>()
                        .join(", "),
                    aggs.join(", "),
                    cost,
                    rows,
                    child
                )
            }
            PlanNode::Join {
                left,
                right,
                join_type,
                condition,
            } => {
                let left_str = self.format_plan(left, indent + 1);
                let right_str = self.format_plan(right, indent + 1);
                let cond = condition
                    .as_ref()
                    .map(|c| format!(" ON {:?}", c))
                    .unwrap_or_default();
                format!(
                    "{}Join {:?}{} (cost={:.2}, rows={})\n{}\n{}",
                    prefix, join_type, cond, cost, rows, left_str, right_str
                )
            }
            PlanNode::Window {
                input,
                window_functions,
            } => {
                let child = self.format_plan(input, indent + 1);
                let funcs: Vec<_> = window_functions
                    .iter()
                    .map(|wf| format!("{} AS {}", wf.function, wf.alias))
                    .collect();
                format!(
                    "{}Window [{}] (cost={:.2}, rows={})\n{}",
                    prefix,
                    funcs.join(", "),
                    cost,
                    rows,
                    child
                )
            }
            PlanNode::SetOp { left, right, op } => {
                let left_str = self.format_plan(left, indent + 1);
                let right_str = self.format_plan(right, indent + 1);
                let op_name = match op {
                    SetOperationType::Union => "Union",
                    SetOperationType::UnionAll => "Union All",
                    SetOperationType::Except => "Except",
                    SetOperationType::ExceptAll => "Except All",
                    SetOperationType::Intersect => "Intersect",
                    SetOperationType::IntersectAll => "Intersect All",
                };
                format!(
                    "{}{} (cost={:.2}, rows={})\n{}\n{}",
                    prefix, op_name, cost, rows, left_str, right_str
                )
            }
            PlanNode::Distinct { input } => {
                let child = self.format_plan(input, indent + 1);
                format!(
                    "{}Distinct (cost={:.2}, rows={})\n{}",
                    prefix, cost, rows, child
                )
            }
            PlanNode::RecursiveCte {
                name,
                base,
                recursive,
                ..
            } => {
                let base_str = self.format_plan(base, indent + 1);
                let rec_str = self.format_plan(recursive, indent + 1);
                format!(
                    "{}RecursiveCTE {} (cost={:.2}, rows={})\n{}  Base:\n{}\n{}  Recursive:\n{}",
                    prefix, name, cost, rows, prefix, base_str, prefix, rec_str
                )
            }
            PlanNode::VectorScan {
                table,
                vector_column,
                metric,
                k,
                filter,
                ..
            } => {
                let filter_str = filter
                    .as_ref()
                    .map(|f| format!(" WHERE {:?}", f))
                    .unwrap_or_default();
                format!(
                    "{}VectorScan {} on {} [{:?}, k={}]{} (cost={:.2}, rows={})",
                    prefix, table, vector_column, metric, k, filter_str, cost, rows
                )
            }
            PlanNode::WcojJoin {
                atoms,
                output_variables,
            } => {
                format!(
                    "{}WcojJoin atoms={} output=[{}] (cost={:.2}, rows={})",
                    prefix,
                    atoms.len(),
                    output_variables.join(", "),
                    cost,
                    rows
                )
            }
            PlanNode::Empty => format!("{}Empty (cost=0, rows=0)", prefix),
        }
    }

    fn build_plan(&self, query: &Query, parent_ctes: &[crate::ast::Cte]) -> QueryResult<PlanNode> {
        // Merge parent CTEs and local CTEs
        // Local CTEs shadow parent CTEs
        let mut all_ctes = parent_ctes.to_vec();
        all_ctes.extend_from_slice(&query.ctes);

        match query.query_type {
            QueryType::Select => self.build_select_plan(query, &all_ctes),
            QueryType::Insert => self.build_insert_plan(query),
            QueryType::Update => self.build_update_plan(query),
            QueryType::Delete => self.build_delete_plan(query),
            _ => Err(QueryError::Unsupported(format!(
                "Query type {:?} not yet supported",
                query.query_type
            ))),
        }
    }

    fn build_select_plan(&self, query: &Query, ctes: &[crate::ast::Cte]) -> QueryResult<PlanNode> {
        // Handle SELECT without FROM — produce single empty row for expression evaluation
        let table = match query.source.as_ref() {
            Some(t) => t,
            None => {
                // SELECT without FROM: use Empty node — executor handles derived_columns
                let mut plan = PlanNode::Empty;
                if !query.columns.is_empty() {
                    plan = plan.project(query.columns.clone());
                }
                if query.distinct {
                    plan = PlanNode::Distinct {
                        input: Box::new(plan),
                    };
                }
                if query.limit.is_some() || query.offset.is_some() {
                    plan = plan.limit(query.limit, query.offset);
                }
                return Ok(plan);
            }
        };

        // Normalize columns: if "*" is present, treat as empty (all columns)
        let effective_columns = if query.columns.iter().any(|c| c == "*") {
            Vec::new()
        } else {
            query.columns.clone()
        };

        // Detect window functions early so we can exclude their aliases from scan
        let mut window_fns = Vec::new();
        let mut window_aliases = std::collections::HashSet::new();
        for (alias, expr) in &query.derived_columns {
            if let Expression::WindowFunction {
                function,
                args,
                window,
            } = expr
            {
                window_fns.push(WindowFunctionDef {
                    alias: alias.clone(),
                    function: function.clone(),
                    args: args.clone(),
                    window: window.clone(),
                });
                window_aliases.insert(alias.clone());
            }
        }

        // For scan/projection, exclude window function aliases (they don't exist in the table)
        let scan_columns: Vec<String> = if window_aliases.is_empty() {
            effective_columns.clone()
        } else {
            effective_columns
                .iter()
                .filter(|c| !window_aliases.contains(*c))
                .cloned()
                .collect()
        };

        // Check if table is a CTE
        let mut plan = if let Some(cte) = ctes.iter().find(|c| c.name == *table) {
            // Build plan for the CTE (recursive CTEs are handled at query level,
            // not at the plan level — the planner builds the base plan)
            let mut plan = self.build_plan(&cte.query, ctes)?;

            // For CTEs, we can't "push down" the projection effectively since it's a separate query.
            // So we must apply the projection here if specific columns are requested.
            if !scan_columns.is_empty() {
                plan = plan.project(scan_columns.clone());
            }
            plan
        } else {
            PlanNode::Scan {
                table: table.clone(),
                columns: if self.projection_pushdown {
                    scan_columns.clone()
                } else {
                    Vec::new()
                },
                filter: if self.predicate_pushdown {
                    query.filter.clone()
                } else {
                    None
                },
            }
        };

        // Add joins
        for join in &query.joins {
            let right = if let Some(cte) = ctes.iter().find(|c| c.name == join.table) {
                self.build_plan(&cte.query, ctes)?
            } else {
                PlanNode::Scan {
                    table: join.table.clone(),
                    columns: Vec::new(),
                    filter: None,
                }
            };
            plan = PlanNode::Join {
                left: Box::new(plan),
                right: Box::new(right),
                join_type: join.join_type,
                condition: join.condition.clone(),
            };
        }

        // Add filter if not pushed down
        if !self.predicate_pushdown {
            if let Some(filter) = &query.filter {
                plan = plan.filter(filter.clone());
            }
        }

        // Add projection if not pushed down (use scan_columns to exclude window aliases)
        if !self.projection_pushdown && !scan_columns.is_empty() {
            plan = plan.project(scan_columns.clone());
        }

        // Add window functions (already detected earlier)
        if !window_fns.is_empty() {
            plan = PlanNode::Window {
                input: Box::new(plan),
                window_functions: window_fns,
            };
        }

        // Add group by
        if !query.group_by.is_empty() {
            plan = PlanNode::Aggregate {
                input: Box::new(plan),
                group_by: query.group_by.clone(),
                aggregates: Vec::new(),
            };
        }

        // Add having
        if let Some(having) = &query.having {
            plan = plan.filter(having.clone());
        }

        // Add distinct
        if query.distinct {
            plan = PlanNode::Distinct {
                input: Box::new(plan),
            };
        }

        // Add order by
        if !query.order_by.is_empty() {
            plan = plan.sort(query.order_by.clone());
        }

        // Add limit/offset
        if query.limit.is_some() || query.offset.is_some() {
            plan = plan.limit(query.limit, query.offset);
        }

        Ok(plan)
    }

    fn build_insert_plan(&self, _query: &Query) -> QueryResult<PlanNode> {
        // INSERT doesn't produce a scan-based plan
        Ok(PlanNode::Empty)
    }

    fn build_update_plan(&self, query: &Query) -> QueryResult<PlanNode> {
        // UPDATE starts with a scan to find matching rows
        let table = query
            .source
            .as_ref()
            .ok_or_else(|| QueryError::ParseError("No table specified".to_string()))?;

        let mut plan = PlanNode::scan(table);

        if let Some(filter) = &query.filter {
            plan = plan.filter(filter.clone());
        }

        Ok(plan)
    }

    fn build_delete_plan(&self, query: &Query) -> QueryResult<PlanNode> {
        // DELETE starts with a scan to find matching rows
        let table = query
            .source
            .as_ref()
            .ok_or_else(|| QueryError::ParseError("No table specified".to_string()))?;

        let mut plan = PlanNode::scan(table);

        if let Some(filter) = &query.filter {
            plan = plan.filter(filter.clone());
        }

        Ok(plan)
    }

    fn optimize(&self, plan: PlanNode) -> QueryResult<PlanNode> {
        if !self.cost_based {
            return Ok(plan);
        }

        // Apply optimization rules in order
        let mut plan = plan;

        // Rule 1: Predicate pushdown - push filters closer to scans
        plan = self.push_down_predicates(plan);

        // Rule 2: Projection pushdown - push projections down
        plan = self.push_down_projections(plan);

        // Rule 3: Eliminate redundant projections
        plan = self.eliminate_redundant_projections(plan);

        // Rule 4: Merge adjacent filters
        plan = self.merge_filters(plan);

        // Rule 5: Join reordering - put smaller tables first
        if self.join_reorder {
            plan = self.reorder_joins(plan);
        }

        // Rule 6: Constant folding
        plan = self.fold_constants(plan);

        // Rule 7: Index selection
        if self.use_indexes {
            plan = self.select_indexes(plan);
        }

        Ok(plan)
    }

    /// Push predicates down to scans
    fn push_down_predicates(&self, plan: PlanNode) -> PlanNode {
        match plan {
            PlanNode::Filter { input, predicate } => {
                let input = self.push_down_predicates(*input);

                // Try to push into scan
                if let PlanNode::Scan {
                    table,
                    columns,
                    filter,
                } = input
                {
                    let combined = match filter {
                        Some(existing) => Expression::and(existing, predicate),
                        None => predicate,
                    };
                    return PlanNode::Scan {
                        table,
                        columns,
                        filter: Some(combined),
                    };
                }

                // Try to push through join
                if let PlanNode::Join {
                    left,
                    right,
                    join_type,
                    condition,
                } = input
                {
                    // Check if predicate only references left or right table
                    // For simplicity, keep filter above join for now
                    return PlanNode::Filter {
                        input: Box::new(PlanNode::Join {
                            left,
                            right,
                            join_type,
                            condition,
                        }),
                        predicate,
                    };
                }

                PlanNode::Filter {
                    input: Box::new(input),
                    predicate,
                }
            }
            PlanNode::Join {
                left,
                right,
                join_type,
                condition,
            } => PlanNode::Join {
                left: Box::new(self.push_down_predicates(*left)),
                right: Box::new(self.push_down_predicates(*right)),
                join_type,
                condition,
            },
            PlanNode::Sort { input, order_by } => PlanNode::Sort {
                input: Box::new(self.push_down_predicates(*input)),
                order_by,
            },
            PlanNode::Project { input, columns } => PlanNode::Project {
                input: Box::new(self.push_down_predicates(*input)),
                columns,
            },
            PlanNode::Limit {
                input,
                limit,
                offset,
            } => PlanNode::Limit {
                input: Box::new(self.push_down_predicates(*input)),
                limit,
                offset,
            },
            PlanNode::Aggregate {
                input,
                group_by,
                aggregates,
            } => PlanNode::Aggregate {
                input: Box::new(self.push_down_predicates(*input)),
                group_by,
                aggregates,
            },
            other => other,
        }
    }

    /// Push projections down to reduce data volume
    fn push_down_projections(&self, plan: PlanNode) -> PlanNode {
        match plan {
            PlanNode::Project { input, columns } => {
                let input = self.push_down_projections(*input);

                // Try to push into scan
                if let PlanNode::Scan {
                    table,
                    columns: _,
                    filter,
                } = input
                {
                    return PlanNode::Scan {
                        table,
                        columns,
                        filter,
                    };
                }

                PlanNode::Project {
                    input: Box::new(input),
                    columns,
                }
            }
            PlanNode::Filter { input, predicate } => PlanNode::Filter {
                input: Box::new(self.push_down_projections(*input)),
                predicate,
            },
            PlanNode::Sort { input, order_by } => PlanNode::Sort {
                input: Box::new(self.push_down_projections(*input)),
                order_by,
            },
            PlanNode::Join {
                left,
                right,
                join_type,
                condition,
            } => PlanNode::Join {
                left: Box::new(self.push_down_projections(*left)),
                right: Box::new(self.push_down_projections(*right)),
                join_type,
                condition,
            },
            PlanNode::Limit {
                input,
                limit,
                offset,
            } => PlanNode::Limit {
                input: Box::new(self.push_down_projections(*input)),
                limit,
                offset,
            },
            PlanNode::Aggregate {
                input,
                group_by,
                aggregates,
            } => PlanNode::Aggregate {
                input: Box::new(self.push_down_projections(*input)),
                group_by,
                aggregates,
            },
            other => other,
        }
    }

    /// Reorder joins to put smaller tables on the left
    fn reorder_joins(&self, plan: PlanNode) -> PlanNode {
        match plan {
            PlanNode::Join {
                left,
                right,
                join_type,
                condition,
            } => {
                let left = Box::new(self.reorder_joins(*left));
                let right = Box::new(self.reorder_joins(*right));

                let left_rows = left.estimate_rows_with_stats(&self.statistics);
                let right_rows = right.estimate_rows_with_stats(&self.statistics);

                // For inner/cross joins, put smaller table on left (build side for hash join)
                if matches!(join_type, JoinType::Inner | JoinType::Cross) && right_rows < left_rows
                {
                    // Swap and adjust condition
                    let swapped_condition = condition.map(|c| self.swap_condition_sides(c));
                    PlanNode::Join {
                        left: right,
                        right: left,
                        join_type,
                        condition: swapped_condition,
                    }
                } else {
                    PlanNode::Join {
                        left,
                        right,
                        join_type,
                        condition,
                    }
                }
            }
            PlanNode::Filter { input, predicate } => PlanNode::Filter {
                input: Box::new(self.reorder_joins(*input)),
                predicate,
            },
            PlanNode::Sort { input, order_by } => PlanNode::Sort {
                input: Box::new(self.reorder_joins(*input)),
                order_by,
            },
            PlanNode::Project { input, columns } => PlanNode::Project {
                input: Box::new(self.reorder_joins(*input)),
                columns,
            },
            PlanNode::Limit {
                input,
                limit,
                offset,
            } => PlanNode::Limit {
                input: Box::new(self.reorder_joins(*input)),
                limit,
                offset,
            },
            PlanNode::Aggregate {
                input,
                group_by,
                aggregates,
            } => PlanNode::Aggregate {
                input: Box::new(self.reorder_joins(*input)),
                group_by,
                aggregates,
            },
            other => other,
        }
    }

    fn swap_condition_sides(&self, expr: Expression) -> Expression {
        match expr {
            Expression::Binary { op, left, right } => {
                let swapped_op = match op {
                    crate::ast::Operator::Lt => crate::ast::Operator::Gt,
                    crate::ast::Operator::Le => crate::ast::Operator::Ge,
                    crate::ast::Operator::Gt => crate::ast::Operator::Lt,
                    crate::ast::Operator::Ge => crate::ast::Operator::Le,
                    other => other,
                };
                Expression::Binary {
                    op: swapped_op,
                    left: right,
                    right: left,
                }
            }
            other => other,
        }
    }

    /// Fold constant expressions
    fn fold_constants(&self, plan: PlanNode) -> PlanNode {
        match plan {
            PlanNode::Filter { input, predicate } => {
                let folded = self.fold_expression(predicate);

                // If predicate is always true, eliminate filter
                if matches!(folded, Expression::Literal(crate::ast::Value::Bool(true))) {
                    return self.fold_constants(*input);
                }

                // If predicate is always false, return empty
                if matches!(folded, Expression::Literal(crate::ast::Value::Bool(false))) {
                    return PlanNode::Empty;
                }

                PlanNode::Filter {
                    input: Box::new(self.fold_constants(*input)),
                    predicate: folded,
                }
            }
            PlanNode::Scan {
                table,
                columns,
                filter,
            } => PlanNode::Scan {
                table,
                columns,
                filter: filter.map(|f| self.fold_expression(f)),
            },
            PlanNode::Join {
                left,
                right,
                join_type,
                condition,
            } => PlanNode::Join {
                left: Box::new(self.fold_constants(*left)),
                right: Box::new(self.fold_constants(*right)),
                join_type,
                condition: condition.map(|c| self.fold_expression(c)),
            },
            PlanNode::Sort { input, order_by } => PlanNode::Sort {
                input: Box::new(self.fold_constants(*input)),
                order_by,
            },
            PlanNode::Project { input, columns } => PlanNode::Project {
                input: Box::new(self.fold_constants(*input)),
                columns,
            },
            PlanNode::Limit {
                input,
                limit,
                offset,
            } => PlanNode::Limit {
                input: Box::new(self.fold_constants(*input)),
                limit,
                offset,
            },
            PlanNode::Aggregate {
                input,
                group_by,
                aggregates,
            } => PlanNode::Aggregate {
                input: Box::new(self.fold_constants(*input)),
                group_by,
                aggregates,
            },
            other => other,
        }
    }

    fn fold_expression(&self, expr: Expression) -> Expression {
        match expr {
            Expression::Binary { op, left, right } => {
                let left = self.fold_expression(*left);
                let right = self.fold_expression(*right);

                // Try to evaluate constant expressions
                match (&left, &right, &op) {
                    (
                        Expression::Literal(crate::ast::Value::Int(a)),
                        Expression::Literal(crate::ast::Value::Int(b)),
                        crate::ast::Operator::Add,
                    ) => Expression::Literal(crate::ast::Value::Int(a + b)),
                    (
                        Expression::Literal(crate::ast::Value::Int(a)),
                        Expression::Literal(crate::ast::Value::Int(b)),
                        crate::ast::Operator::Sub,
                    ) => Expression::Literal(crate::ast::Value::Int(a - b)),
                    (
                        Expression::Literal(crate::ast::Value::Int(a)),
                        Expression::Literal(crate::ast::Value::Int(b)),
                        crate::ast::Operator::Mul,
                    ) => Expression::Literal(crate::ast::Value::Int(a * b)),
                    (
                        Expression::Literal(crate::ast::Value::Bool(a)),
                        Expression::Literal(crate::ast::Value::Bool(b)),
                        crate::ast::Operator::And,
                    ) => Expression::Literal(crate::ast::Value::Bool(*a && *b)),
                    (
                        Expression::Literal(crate::ast::Value::Bool(a)),
                        Expression::Literal(crate::ast::Value::Bool(b)),
                        crate::ast::Operator::Or,
                    ) => Expression::Literal(crate::ast::Value::Bool(*a || *b)),
                    _ => Expression::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                }
            }
            Expression::Unary { op, expr } => {
                let inner = self.fold_expression(*expr);
                match (&inner, &op) {
                    (
                        Expression::Literal(crate::ast::Value::Bool(b)),
                        crate::ast::UnaryOperator::Not,
                    ) => Expression::Literal(crate::ast::Value::Bool(!b)),
                    (
                        Expression::Literal(crate::ast::Value::Int(n)),
                        crate::ast::UnaryOperator::Neg,
                    ) => Expression::Literal(crate::ast::Value::Int(-n)),
                    _ => Expression::Unary {
                        op,
                        expr: Box::new(inner),
                    },
                }
            }
            other => other,
        }
    }

    /// Select indexes for scans based on predicates
    fn select_indexes(&self, plan: PlanNode) -> PlanNode {
        match plan {
            PlanNode::Scan {
                table,
                columns,
                filter,
            } => {
                // Check if an index can be used
                if let Some(filter_expr) = &filter {
                    // B-tree / Hash indexes
                    if let Some(index) = self.statistics.can_use_index(&table, filter_expr) {
                        return PlanNode::IndexScan {
                            table,
                            index: index.name.clone(),
                            columns,
                            filter: filter_expr.clone(),
                        };
                    }
                    // GIN indexes for @>, <@, LIKE (trigram)
                    if let Some(index) = self.statistics.can_use_gin_index(&table, filter_expr) {
                        return PlanNode::IndexScan {
                            table,
                            index: index.name.clone(),
                            columns,
                            filter: filter_expr.clone(),
                        };
                    }
                }
                PlanNode::Scan {
                    table,
                    columns,
                    filter,
                }
            }
            PlanNode::Filter { input, predicate } => PlanNode::Filter {
                input: Box::new(self.select_indexes(*input)),
                predicate,
            },
            PlanNode::Join {
                left,
                right,
                join_type,
                condition,
            } => PlanNode::Join {
                left: Box::new(self.select_indexes(*left)),
                right: Box::new(self.select_indexes(*right)),
                join_type,
                condition,
            },
            PlanNode::Sort { input, order_by } => PlanNode::Sort {
                input: Box::new(self.select_indexes(*input)),
                order_by,
            },
            PlanNode::Project { input, columns } => PlanNode::Project {
                input: Box::new(self.select_indexes(*input)),
                columns,
            },
            PlanNode::Limit {
                input,
                limit,
                offset,
            } => PlanNode::Limit {
                input: Box::new(self.select_indexes(*input)),
                limit,
                offset,
            },
            PlanNode::Aggregate {
                input,
                group_by,
                aggregates,
            } => PlanNode::Aggregate {
                input: Box::new(self.select_indexes(*input)),
                group_by,
                aggregates,
            },
            other => other,
        }
    }

    fn eliminate_redundant_projections(&self, plan: PlanNode) -> PlanNode {
        match plan {
            PlanNode::Project { input, columns } => {
                let input = self.eliminate_redundant_projections(*input);
                // If projecting all columns from a scan, eliminate
                if let PlanNode::Scan { table, filter, .. } = &input {
                    if columns.iter().any(|c| c == "*") {
                        return PlanNode::Scan {
                            table: table.clone(),
                            columns: Vec::new(),
                            filter: filter.clone(),
                        };
                    }
                }
                PlanNode::Project {
                    input: Box::new(input),
                    columns,
                }
            }
            PlanNode::Filter { input, predicate } => PlanNode::Filter {
                input: Box::new(self.eliminate_redundant_projections(*input)),
                predicate,
            },
            PlanNode::Sort { input, order_by } => PlanNode::Sort {
                input: Box::new(self.eliminate_redundant_projections(*input)),
                order_by,
            },
            PlanNode::Limit {
                input,
                limit,
                offset,
            } => PlanNode::Limit {
                input: Box::new(self.eliminate_redundant_projections(*input)),
                limit,
                offset,
            },
            other => other,
        }
    }

    fn merge_filters(&self, plan: PlanNode) -> PlanNode {
        match plan {
            PlanNode::Filter {
                input,
                predicate: outer,
            } => {
                let input = self.merge_filters(*input);
                if let PlanNode::Filter {
                    input: inner_input,
                    predicate: inner,
                } = input
                {
                    // Merge filters with AND
                    PlanNode::Filter {
                        input: inner_input,
                        predicate: Expression::and(inner, outer),
                    }
                } else {
                    PlanNode::Filter {
                        input: Box::new(input),
                        predicate: outer,
                    }
                }
            }
            other => other,
        }
    }
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Value;

    #[test]
    fn test_plan_simple_select() {
        let planner = QueryPlanner::new();
        let query = Query::select("users").columns(vec!["id", "name"]);

        let plan = planner.plan(&query).unwrap();
        assert!(plan.estimated_rows > 0);
    }

    #[test]
    fn test_plan_with_filter() {
        let planner = QueryPlanner::new();
        let query = Query::select("users").filter(Expression::eq(
            Expression::column("id"),
            Expression::literal(Value::Int(1)),
        ));

        let plan = planner.plan(&query).unwrap();
        // Filter should reduce estimated rows
        assert!(plan.estimated_rows < 1000);
    }

    #[test]
    fn test_plan_with_limit() {
        let planner = QueryPlanner::new();
        let query = Query::select("users").limit(10);

        let plan = planner.plan(&query).unwrap();
        assert!(plan.estimated_rows <= 10);
    }

    #[test]
    fn test_plan_node_cost() {
        let scan = PlanNode::scan("users");
        let filtered = scan.clone().filter(Expression::column("id"));

        // Filter adds CPU cost for processing but reduces output rows
        // Both costs should be positive
        assert!(scan.estimate_cost() > 0.0);
        assert!(filtered.estimate_cost() > 0.0);

        // Filter should reduce estimated rows
        assert!(filtered.estimate_rows() < scan.estimate_rows());
    }

    #[test]
    fn test_plan_node_builder() {
        let plan = PlanNode::scan("users")
            .filter(Expression::column("active"))
            .project(vec!["id".to_string(), "name".to_string()])
            .sort(vec![OrderBy {
                expr: Expression::Column("name".to_string()),
                descending: false,
                nulls_first: None,
            }])
            .limit(Some(10), None);

        match plan {
            PlanNode::Limit { limit, .. } => assert_eq!(limit, Some(10)),
            _ => panic!("Expected Limit node"),
        }
    }

    #[test]
    fn test_statistics_selectivity() {
        let stats = Statistics::new();

        // Equality predicate
        let eq_pred = Expression::eq(Expression::column("id"), Expression::literal(Value::Int(1)));
        let selectivity = stats.selectivity(Some(&eq_pred));
        assert!(selectivity > 0.0 && selectivity < 1.0);

        // No predicate = full selectivity
        assert_eq!(stats.selectivity(None), 1.0);
    }

    #[test]
    fn test_statistics_with_table_stats() {
        let mut stats = Statistics::new();
        stats.add_table(
            "users",
            TableStats {
                row_count: 10000,
                columns: std::collections::HashMap::new(),
                indexes: Vec::new(),
            },
        );

        assert_eq!(stats.table_rows("users"), 10000);
        assert_eq!(stats.table_rows("unknown"), 1000); // Default
    }

    #[test]
    fn test_planner_with_statistics() {
        let mut stats = Statistics::new();
        stats.add_table(
            "small_table",
            TableStats {
                row_count: 100,
                ..Default::default()
            },
        );
        stats.add_table(
            "large_table",
            TableStats {
                row_count: 1000000,
                ..Default::default()
            },
        );

        let planner = QueryPlanner::with_statistics(stats);

        let small_query = Query::select("small_table");
        let large_query = Query::select("large_table");

        let small_plan = planner.plan(&small_query).unwrap();
        let large_plan = planner.plan(&large_query).unwrap();

        assert!(small_plan.estimated_cost < large_plan.estimated_cost);
        assert!(small_plan.estimated_rows < large_plan.estimated_rows);
    }

    #[test]
    fn test_explain_plan() {
        let planner = QueryPlanner::new();
        let query = Query::select("users")
            .columns(vec!["id", "name"])
            .filter(Expression::eq(
                Expression::column("active"),
                Expression::literal(Value::Bool(true)),
            ))
            .limit(10);

        let explain = planner.explain(&query).unwrap();
        assert!(explain.contains("Scan"));
        assert!(explain.contains("users"));
    }

    #[test]
    fn test_cascade_tier_classification() {
        use crate::ast::Operator;

        // Order is energy-ascending.
        assert!(CascadeTier::Lookup < CascadeTier::Formula);
        assert!(CascadeTier::Formula < CascadeTier::Extract);
        assert!(CascadeTier::Extract < CascadeTier::Aggregate);
        assert!(CascadeTier::Aggregate < CascadeTier::Reason);

        // Empty plan — trivially Formula.
        assert_eq!(PlanNode::Empty.cascade_tier(), CascadeTier::Formula);

        // Range scan — Extract.
        let range_scan = PlanNode::Scan {
            table: "users".into(),
            columns: vec![],
            filter: Some(Expression::Binary {
                left: Box::new(Expression::Column("age".into())),
                op: Operator::Gt,
                right: Box::new(Expression::Literal(Value::Int(25))),
            }),
        };
        assert_eq!(range_scan.cascade_tier(), CascadeTier::Extract);

        // Unique-row equality scan — Lookup.
        let pk_scan = PlanNode::Scan {
            table: "users".into(),
            columns: vec![],
            filter: Some(Expression::Binary {
                left: Box::new(Expression::Column("id".into())),
                op: Operator::Eq,
                right: Box::new(Expression::Literal(Value::Int(42))),
            }),
        };
        assert_eq!(pk_scan.cascade_tier(), CascadeTier::Lookup);

        // No-filter scan — Extract.
        let full_scan = PlanNode::Scan {
            table: "users".into(),
            columns: vec![],
            filter: None,
        };
        assert_eq!(full_scan.cascade_tier(), CascadeTier::Extract);

        // Aggregate over scan — Aggregate.
        let agg = PlanNode::Aggregate {
            input: Box::new(full_scan.clone()),
            group_by: vec![],
            aggregates: vec![("c".into(), "COUNT".into(), "*".into())],
        };
        assert_eq!(agg.cascade_tier(), CascadeTier::Aggregate);

        // Join — Aggregate (max of children's tier and Join's own Aggregate).
        let join = PlanNode::Join {
            left: Box::new(pk_scan.clone()),
            right: Box::new(full_scan.clone()),
            join_type: JoinType::Inner,
            condition: None,
        };
        assert_eq!(join.cascade_tier(), CascadeTier::Aggregate);

        // Pure projection over PK scan — tier stays Lookup (Project is pass-through).
        let pk_project = PlanNode::Project {
            input: Box::new(pk_scan.clone()),
            columns: vec!["name".into()],
        };
        assert_eq!(pk_project.cascade_tier(), CascadeTier::Lookup);

        // Limit over scan keeps the input's tier.
        let limited = PlanNode::Limit {
            input: Box::new(full_scan.clone()),
            limit: Some(10),
            offset: None,
        };
        assert_eq!(limited.cascade_tier(), CascadeTier::Extract);

        // Recursive CTE — Reason regardless of children's tier.
        let recursive = PlanNode::RecursiveCte {
            name: "reach".into(),
            base: Box::new(full_scan.clone()),
            recursive: Box::new(full_scan.clone()),
            columns: vec!["src".into(), "dst".into()],
        };
        assert_eq!(recursive.cascade_tier(), CascadeTier::Reason);

        // Window function — Aggregate.
        let window = PlanNode::Window {
            input: Box::new(full_scan.clone()),
            window_functions: vec![],
        };
        assert_eq!(window.cascade_tier(), CascadeTier::Aggregate);

        // Vector scan — Extract (k-NN via index).
        let vec_scan = PlanNode::VectorScan {
            table: "articles".into(),
            vector_column: "embedding".into(),
            query_vector: vec![0.1, 0.2, 0.3],
            metric: VectorMetric::Cosine,
            k: 10,
            filter: None,
        };
        assert_eq!(vec_scan.cascade_tier(), CascadeTier::Extract);

        // Display formatting.
        assert_eq!(CascadeTier::Lookup.to_string(), "Lookup");
        assert_eq!(CascadeTier::Reason.to_string(), "Reason");
    }

    #[test]
    fn test_constant_folding() {
        let planner = QueryPlanner::new();

        // Build a plan with a constant expression
        let const_expr = Expression::Binary {
            op: crate::ast::Operator::Add,
            left: Box::new(Expression::Literal(Value::Int(1))),
            right: Box::new(Expression::Literal(Value::Int(2))),
        };

        let folded = planner.fold_expression(const_expr);
        assert_eq!(folded, Expression::Literal(Value::Int(3)));
    }

    #[test]
    fn test_join_cost_estimation() {
        let left = PlanNode::scan("users");
        let right = PlanNode::scan("orders");

        let join = PlanNode::Join {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            condition: Some(Expression::eq(
                Expression::column("users.id"),
                Expression::column("orders.user_id"),
            )),
        };

        let stats = Statistics::default();
        let cost = join.estimate_cost_with_stats(&stats);
        let rows = join.estimate_rows_with_stats(&stats);

        assert!(cost > 0.0);
        assert!(rows > 0);
    }
}
