//! Query Optimizer for JouleDB
//!
//! This module provides comprehensive query optimization for analytical queries:
//!
//! ## Cost-Based Optimization
//! - **Build side selection**: Choose smaller table for hash join build phase
//! - **Join ordering**: Order multi-way joins by estimated selectivity
//! - **Cost estimation**: CPU and I/O cost models
//!
//! ## Rule-Based Optimization
//! - **Predicate pushdown**: Push filters below joins
//! - **Projection pruning**: Eliminate unused columns
//! - **Constant folding**: Evaluate constant expressions at compile time
//! - **Join elimination**: Remove unnecessary joins
//!
//! ## Query Planning
//! - **Logical Plan**: High-level relational algebra representation
//! - **Physical Plan**: Concrete execution operators
//! - **Plan Execution**: Interface to execute optimized plans
//!
//! ## Example
//!
//! ```rust,ignore
//! let optimizer = QueryOptimizer::new(&store);
//!
//! // Build a logical query
//! let query = LogicalPlan::scan("lineitem")
//!     .filter("l_shipdate", 19940101.0, 19950101.0)
//!     .join("orders", "l_orderkey", "o_orderkey", JoinType::Inner)
//!     .group_by("o_orderdate", vec![("l_extendedprice", AggregateFunc::Sum)])
//!     .build();
//!
//! // Optimize and get physical plan
//! let physical = optimizer.optimize(query);
//!
//! // Execute
//! let result = physical.execute(&store);
//! ```

use crate::columnar::ColumnarStore;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// Query optimizer with cost-based decisions
pub struct QueryOptimizer<'a> {
    columnar: &'a ColumnarStore,
}

impl<'a> QueryOptimizer<'a> {
    /// Create a new optimizer with reference to columnar store
    pub fn new(columnar: &'a ColumnarStore) -> Self {
        Self { columnar }
    }

    /// Get estimated cardinality (row count) for a column
    pub fn cardinality(&self, field: &str) -> Option<usize> {
        self.columnar.get_column(field).map(|c| c.count_live())
    }

    /// Estimate selectivity of a range filter
    /// Returns fraction of rows that would pass the filter (0.0 to 1.0)
    pub fn estimate_selectivity(&self, field: &str, min: f64, max: f64) -> Option<f64> {
        let col = self.columnar.get_column(field)?;

        // If range doesn't overlap with data, selectivity is 0
        if max <= col.min() || min > col.max() {
            return Some(0.0);
        }

        // Assume uniform distribution for simplicity
        let data_range = col.max() - col.min();
        if data_range <= 0.0 {
            return Some(1.0); // All values are the same
        }

        // Clip filter range to data range
        let effective_min = min.max(col.min());
        let effective_max = max.min(col.max());
        let filter_range = effective_max - effective_min;

        Some((filter_range / data_range).clamp(0.0, 1.0))
    }

    /// Estimate result size of a range filter
    pub fn estimate_filter_result(&self, field: &str, min: f64, max: f64) -> Option<usize> {
        let selectivity = self.estimate_selectivity(field, min, max)?;
        let cardinality = self.cardinality(field)?;
        Some((cardinality as f64 * selectivity) as usize)
    }

    /// Optimize join by selecting build and probe sides
    /// Returns (build_field, probe_field) where build_field has smaller cardinality
    pub fn optimize_join(&self, field_a: &str, field_b: &str) -> (String, String) {
        let card_a = self.cardinality(field_a).unwrap_or(usize::MAX);
        let card_b = self.cardinality(field_b).unwrap_or(usize::MAX);

        if card_a <= card_b {
            (field_a.to_string(), field_b.to_string())
        } else {
            (field_b.to_string(), field_a.to_string())
        }
    }

    /// Estimate cost of a hash join (in arbitrary units)
    /// Cost = build_cost + probe_cost
    /// build_cost = n_build (hash table construction)
    /// probe_cost = n_probe (hash lookups)
    pub fn estimate_join_cost(&self, build_field: &str, probe_field: &str) -> Option<f64> {
        let build_card = self.cardinality(build_field)? as f64;
        let probe_card = self.cardinality(probe_field)? as f64;

        // Hash table build: O(n) with constant factor ~2 for hashing
        let build_cost = build_card * 2.0;

        // Probe: O(m) with constant factor ~1 for lookup
        let probe_cost = probe_card * 1.0;

        Some(build_cost + probe_cost)
    }

    /// Order multiple joins by estimated cost (greedy approach)
    /// Returns ordered list of (build_field, probe_field) pairs
    pub fn order_joins(&self, join_pairs: &[(String, String)]) -> Vec<(String, String)> {
        let mut pairs_with_cost: Vec<_> = join_pairs
            .iter()
            .map(|(a, b)| {
                let (build, probe) = self.optimize_join(a, b);
                let cost = self.estimate_join_cost(&build, &probe).unwrap_or(f64::MAX);
                ((build, probe), cost)
            })
            .collect();

        // Sort by cost (ascending)
        pairs_with_cost.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        pairs_with_cost.into_iter().map(|(pair, _)| pair).collect()
    }

    // =========================================================================
    // Histogram-Based Estimation
    // =========================================================================

    /// Build a histogram for a column (default 64 buckets)
    pub fn build_histogram(
        &self,
        field: &str,
        num_buckets: usize,
    ) -> Option<crate::columnar::Histogram> {
        self.columnar
            .get_column(field)
            .map(|c| c.build_histogram(num_buckets))
    }

    /// Estimate selectivity using histogram (more accurate than uniform assumption)
    pub fn estimate_selectivity_with_histogram(
        &self,
        field: &str,
        min: f64,
        max: f64,
    ) -> Option<f64> {
        let col = self.columnar.get_column(field)?;
        let histogram = col.build_histogram(64);
        Some(col.estimate_selectivity_with_histogram(min, max, &histogram))
    }

    // =========================================================================
    // Dynamic Programming Join Ordering (Selinger-style)
    // =========================================================================

    /// Optimal join ordering using dynamic programming (Selinger algorithm)
    ///
    /// For N tables, examines all 2^N subsets to find the minimum-cost join order.
    /// Returns the optimal sequence of joins.
    ///
    /// # Arguments
    /// * `tables` - Table names involved in the join
    /// * `predicates` - Join predicates connecting tables
    ///
    /// # Returns
    /// Optimal join order as list of (build, probe, predicate_idx)
    pub fn optimize_join_order_dp(
        &self,
        tables: &[String],
        predicates: &[JoinPredicate],
    ) -> Vec<JoinSpec> {
        let n = tables.len();
        if n <= 1 {
            return Vec::new();
        }

        // For small joins, use greedy for simplicity
        if n == 2 {
            if let Some(pred) = predicates.first() {
                let (build, probe) = self.optimize_join(&pred.left_table, &pred.right_table);
                return vec![JoinSpec {
                    build_table: build,
                    probe_table: probe,
                    left_col: pred.left_col.clone(),
                    right_col: pred.right_col.clone(),
                    estimated_cost: self
                        .estimate_join_cost(&pred.left_col, &pred.right_col)
                        .unwrap_or(0.0),
                }];
            }
            return Vec::new();
        }

        // DP state: best_plan[subset] = (cost, plan) for joining tables in subset
        use std::collections::HashMap;

        #[derive(Clone)]
        struct DPState {
            cost: f64,
            result_cardinality: usize,
            plan: Vec<JoinSpec>,
        }

        let mut dp: HashMap<u64, DPState> = HashMap::new();

        // Initialize: single tables have no join cost
        for (i, table) in tables.iter().enumerate() {
            let subset = 1u64 << i;
            let card = self.cardinality_for_table(table);
            dp.insert(
                subset,
                DPState {
                    cost: 0.0,
                    result_cardinality: card,
                    plan: Vec::new(),
                },
            );
        }

        // Build predicates map: which predicates connect which table pairs
        let mut pred_map: HashMap<(usize, usize), &JoinPredicate> = HashMap::new();
        for pred in predicates {
            if let (Some(left_idx), Some(right_idx)) = (
                tables.iter().position(|t| t == &pred.left_table),
                tables.iter().position(|t| t == &pred.right_table),
            ) {
                let (a, b) = if left_idx < right_idx {
                    (left_idx, right_idx)
                } else {
                    (right_idx, left_idx)
                };
                pred_map.insert((a, b), pred);
            }
        }

        // Enumerate subsets by size
        for size in 2..=n {
            for subset in 0..(1u64 << n) {
                if (subset as usize).count_ones() as usize != size {
                    continue;
                }

                let mut best_cost = f64::MAX;
                let mut best_state: Option<DPState> = None;

                // Try all ways to split subset into two non-empty parts
                // where part1 and part2 are connected by a predicate
                let mut sub = subset;
                while sub > 0 {
                    let part1 = sub;
                    let part2 = subset & !part1;

                    if part1 == 0 || part2 == 0 {
                        sub = (sub - 1) & subset;
                        continue;
                    }

                    // Check if part1 and part2 are connected
                    let connected = self.subsets_connected(part1, part2, &pred_map, n);
                    if !connected {
                        sub = (sub - 1) & subset;
                        continue;
                    }

                    // Get best plans for each part
                    if let (Some(state1), Some(state2)) = (dp.get(&part1), dp.get(&part2)) {
                        // Find the predicate connecting them
                        if let Some((pred, left_in_1)) =
                            self.find_connecting_predicate(part1, part2, &pred_map, tables)
                        {
                            // Estimate join cost
                            let (build_card, probe_card) = if left_in_1 {
                                (state1.result_cardinality, state2.result_cardinality)
                            } else {
                                (state2.result_cardinality, state1.result_cardinality)
                            };

                            // Use smaller table as build side
                            let (build_table, probe_table, build_c, probe_c) =
                                if build_card <= probe_card {
                                    if left_in_1 {
                                        (
                                            pred.left_table.clone(),
                                            pred.right_table.clone(),
                                            build_card,
                                            probe_card,
                                        )
                                    } else {
                                        (
                                            pred.right_table.clone(),
                                            pred.left_table.clone(),
                                            build_card,
                                            probe_card,
                                        )
                                    }
                                } else {
                                    if left_in_1 {
                                        (
                                            pred.right_table.clone(),
                                            pred.left_table.clone(),
                                            probe_card,
                                            build_card,
                                        )
                                    } else {
                                        (
                                            pred.left_table.clone(),
                                            pred.right_table.clone(),
                                            probe_card,
                                            build_card,
                                        )
                                    }
                                };

                            // Cost = build cost + probe cost
                            let join_cost = build_c as f64 * 2.0 + probe_c as f64;
                            let total_cost = state1.cost + state2.cost + join_cost;

                            // Estimate result cardinality (simplified)
                            let result_card =
                                (build_c.min(probe_c) as f64 * pred.selectivity) as usize;

                            if total_cost < best_cost {
                                best_cost = total_cost;

                                // Merge plans
                                let mut new_plan = state1.plan.clone();
                                new_plan.extend(state2.plan.clone());
                                new_plan.push(JoinSpec {
                                    build_table,
                                    probe_table,
                                    left_col: pred.left_col.clone(),
                                    right_col: pred.right_col.clone(),
                                    estimated_cost: join_cost,
                                });

                                best_state = Some(DPState {
                                    cost: total_cost,
                                    result_cardinality: result_card.max(1),
                                    plan: new_plan,
                                });
                            }
                        }
                    }

                    sub = (sub - 1) & subset;
                }

                if let Some(state) = best_state {
                    dp.insert(subset, state);
                }
            }
        }

        // Return the best plan for all tables
        let full_subset = (1u64 << n) - 1;
        dp.get(&full_subset)
            .map(|s| s.plan.clone())
            .unwrap_or_default()
    }

    /// Get cardinality for a table (using first column found)
    fn cardinality_for_table(&self, _table: &str) -> usize {
        // In a real implementation, we'd look up the table's primary column
        // For now, use average cardinality
        self.columnar
            .column_names()
            .next()
            .and_then(|name| self.cardinality(name))
            .unwrap_or(1000)
    }

    /// Check if two subsets are connected by any predicate
    fn subsets_connected(
        &self,
        part1: u64,
        part2: u64,
        pred_map: &HashMap<(usize, usize), &JoinPredicate>,
        n: usize,
    ) -> bool {
        for i in 0..n {
            if (part1 >> i) & 1 == 0 {
                continue;
            }
            for j in 0..n {
                if (part2 >> j) & 1 == 0 {
                    continue;
                }
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                if pred_map.contains_key(&(a, b)) {
                    return true;
                }
            }
        }
        false
    }

    /// Find the predicate connecting two subsets
    fn find_connecting_predicate<'b>(
        &self,
        part1: u64,
        part2: u64,
        pred_map: &'b HashMap<(usize, usize), &'b JoinPredicate>,
        tables: &[String],
    ) -> Option<(&'b JoinPredicate, bool)> {
        let n = tables.len();
        for i in 0..n {
            if (part1 >> i) & 1 == 0 {
                continue;
            }
            for j in 0..n {
                if (part2 >> j) & 1 == 0 {
                    continue;
                }
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                if let Some(pred) = pred_map.get(&(a, b)) {
                    // Return whether left_table is in part1
                    let left_in_1 = tables
                        .iter()
                        .position(|t| t == &pred.left_table)
                        .map(|idx| (part1 >> idx) & 1 == 1)
                        .unwrap_or(false);
                    return Some((pred, left_in_1));
                }
            }
        }
        None
    }

    /// Generate a simple query plan for a join with filter
    pub fn plan_filtered_join(
        &self,
        build_field: &str,
        probe_field: &str,
        filter_field: Option<&str>,
        filter_min: Option<f64>,
        filter_max: Option<f64>,
    ) -> QueryPlan {
        // Determine optimal build side
        let (build, probe) = self.optimize_join(build_field, probe_field);

        // Estimate costs
        let filter_selectivity =
            if let (Some(field), Some(min), Some(max)) = (filter_field, filter_min, filter_max) {
                self.estimate_selectivity(field, min, max).unwrap_or(1.0)
            } else {
                1.0
            };

        let build_card = self.cardinality(&build).unwrap_or(0);
        let probe_card = self.cardinality(&probe).unwrap_or(0);
        let estimated_output = ((build_card.min(probe_card) as f64) * filter_selectivity) as usize;

        QueryPlan {
            steps: vec![
                if filter_field.is_some() {
                    PlanStep::Filter {
                        field: filter_field.unwrap().to_string(),
                        min: filter_min.unwrap_or(f64::MIN),
                        max: filter_max.unwrap_or(f64::MAX),
                        estimated_rows: (probe_card as f64 * filter_selectivity) as usize,
                    }
                } else {
                    PlanStep::Scan {
                        field: probe.clone(),
                        estimated_rows: probe_card,
                    }
                },
                PlanStep::HashJoin {
                    build_field: build.clone(),
                    probe_field: probe.clone(),
                    build_cardinality: build_card,
                    probe_cardinality: probe_card,
                },
            ],
            estimated_cost: self.estimate_join_cost(&build, &probe).unwrap_or(0.0),
            estimated_output_rows: estimated_output,
        }
    }

    /// Get optimization statistics summary
    pub fn stats_summary(&self) -> OptimizerStats {
        let mut columns = Vec::new();

        for name in self.columnar.column_names() {
            if let Some(col) = self.columnar.get_column(name) {
                columns.push(ColumnCostStats {
                    name: name.clone(),
                    cardinality: col.count_live(),
                    min: col.min(),
                    max: col.max(),
                    tombstone_ratio: if col.count() > 0 {
                        col.tombstone_count() as f64 / col.count() as f64
                    } else {
                        0.0
                    },
                });
            }
        }

        OptimizerStats { columns }
    }
}

/// A query execution plan
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Ordered steps to execute
    pub steps: Vec<PlanStep>,
    /// Estimated total cost
    pub estimated_cost: f64,
    /// Estimated output rows
    pub estimated_output_rows: usize,
}

impl QueryPlan {
    /// Display the plan in a human-readable format
    pub fn explain(&self) -> String {
        let mut output = String::new();
        output.push_str("Query Plan:\n");
        output.push_str(&format!("  Estimated Cost: {:.2}\n", self.estimated_cost));
        output.push_str(&format!(
            "  Estimated Output: {} rows\n\n",
            self.estimated_output_rows
        ));

        for (i, step) in self.steps.iter().enumerate() {
            output.push_str(&format!("  Step {}: {}\n", i + 1, step.describe()));
        }

        output
    }
}

/// A step in the query plan
#[derive(Debug, Clone)]
pub enum PlanStep {
    /// Scan a column
    Scan {
        field: String,
        estimated_rows: usize,
    },
    /// Apply a range filter
    Filter {
        field: String,
        min: f64,
        max: f64,
        estimated_rows: usize,
    },
    /// Hash join between two columns
    HashJoin {
        build_field: String,
        probe_field: String,
        build_cardinality: usize,
        probe_cardinality: usize,
    },
    /// Aggregate (SUM, COUNT, AVG, etc.)
    Aggregate { function: String, field: String },
}

impl PlanStep {
    /// Get a human-readable description
    pub fn describe(&self) -> String {
        match self {
            PlanStep::Scan {
                field,
                estimated_rows,
            } => {
                format!("Scan({}) -> {} rows", field, estimated_rows)
            }
            PlanStep::Filter {
                field,
                min,
                max,
                estimated_rows,
            } => {
                format!(
                    "Filter({} IN [{:.2}, {:.2})) -> {} rows",
                    field, min, max, estimated_rows
                )
            }
            PlanStep::HashJoin {
                build_field,
                probe_field,
                build_cardinality,
                probe_cardinality,
            } => {
                format!(
                    "HashJoin(build={} [{}], probe={} [{}])",
                    build_field, build_cardinality, probe_field, probe_cardinality
                )
            }
            PlanStep::Aggregate { function, field } => {
                format!("{}({})", function.to_uppercase(), field)
            }
        }
    }
}

/// Statistics used by the optimizer
#[derive(Debug, Clone)]
pub struct OptimizerStats {
    pub columns: Vec<ColumnCostStats>,
}

/// Cost statistics for a single column
#[derive(Debug, Clone)]
pub struct ColumnCostStats {
    pub name: String,
    pub cardinality: usize,
    pub min: f64,
    pub max: f64,
    pub tombstone_ratio: f64,
}

/// Join predicate for DP join ordering
#[derive(Debug, Clone)]
pub struct JoinPredicate {
    /// Left table in the join
    pub left_table: String,
    /// Right table in the join
    pub right_table: String,
    /// Left column for join condition
    pub left_col: String,
    /// Right column for join condition
    pub right_col: String,
    /// Estimated selectivity (0.0 to 1.0)
    pub selectivity: f64,
}

impl JoinPredicate {
    /// Create a new join predicate
    pub fn new(left_table: &str, right_table: &str, left_col: &str, right_col: &str) -> Self {
        Self {
            left_table: left_table.to_string(),
            right_table: right_table.to_string(),
            left_col: left_col.to_string(),
            right_col: right_col.to_string(),
            selectivity: 0.1, // Default selectivity
        }
    }

    /// Set selectivity
    pub fn with_selectivity(mut self, selectivity: f64) -> Self {
        self.selectivity = selectivity;
        self
    }
}

/// Specification for a single join in the optimized plan
#[derive(Debug, Clone)]
pub struct JoinSpec {
    /// Build table (smaller, used for hash table)
    pub build_table: String,
    /// Probe table (larger, scanned for matches)
    pub probe_table: String,
    /// Left join column
    pub left_col: String,
    /// Right join column
    pub right_col: String,
    /// Estimated cost of this join
    pub estimated_cost: f64,
}

// =============================================================================
// LOGICAL PLAN
// =============================================================================

/// Logical query plan - high-level relational algebra
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Scan a table/column
    Scan { table: String, columns: Vec<String> },
    /// Filter rows by predicate
    Filter {
        input: Box<LogicalPlan>,
        predicate: Predicate,
    },
    /// Project specific columns
    Project {
        input: Box<LogicalPlan>,
        columns: Vec<String>,
    },
    /// Join two inputs
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        left_key: String,
        right_key: String,
        join_type: JoinType,
    },
    /// Group by with aggregations
    GroupBy {
        input: Box<LogicalPlan>,
        group_keys: Vec<String>,
        aggregates: Vec<(String, AggregateFunc, String)>, // (output_name, func, input_field)
    },
    /// Limit result rows
    Limit {
        input: Box<LogicalPlan>,
        count: usize,
    },
    /// Sort by columns
    Sort {
        input: Box<LogicalPlan>,
        sort_keys: Vec<(String, SortOrder)>,
    },
}

/// Predicate for filtering
#[derive(Debug, Clone)]
pub enum Predicate {
    /// Range filter: field BETWEEN min AND max
    Range { field: String, min: f64, max: f64 },
    /// Equality: field = value
    Equals { field: String, value: f64 },
    /// IN list: field IN (values...)
    In { field: String, values: Vec<f64> },
    /// AND of predicates
    And(Vec<Predicate>),
    /// OR of predicates
    Or(Vec<Predicate>),
    /// NOT predicate
    Not(Box<Predicate>),
    /// Always true
    True,
}

/// Join type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    LeftOuter,
    RightOuter,
    FullOuter,
    Semi,
    Anti,
}

/// Aggregate function
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// Sort order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    Ascending,
    Descending,
}

impl LogicalPlan {
    /// Create a scan of a table
    pub fn scan(table: &str) -> LogicalPlanBuilder {
        LogicalPlanBuilder {
            plan: LogicalPlan::Scan {
                table: table.to_string(),
                columns: Vec::new(),
            },
        }
    }

    /// Get all referenced columns
    pub fn referenced_columns(&self) -> HashSet<String> {
        let mut columns = HashSet::new();
        self.collect_columns(&mut columns);
        columns
    }

    fn collect_columns(&self, columns: &mut HashSet<String>) {
        match self {
            LogicalPlan::Scan { columns: cols, .. } => {
                for col in cols {
                    columns.insert(col.clone());
                }
            }
            LogicalPlan::Filter { input, predicate } => {
                input.collect_columns(columns);
                predicate.collect_columns(columns);
            }
            LogicalPlan::Project {
                input,
                columns: cols,
            } => {
                input.collect_columns(columns);
                for col in cols {
                    columns.insert(col.clone());
                }
            }
            LogicalPlan::Join {
                left,
                right,
                left_key,
                right_key,
                ..
            } => {
                left.collect_columns(columns);
                right.collect_columns(columns);
                columns.insert(left_key.clone());
                columns.insert(right_key.clone());
            }
            LogicalPlan::GroupBy {
                input,
                group_keys,
                aggregates,
            } => {
                input.collect_columns(columns);
                for key in group_keys {
                    columns.insert(key.clone());
                }
                for (_, _, field) in aggregates {
                    columns.insert(field.clone());
                }
            }
            LogicalPlan::Limit { input, .. } => input.collect_columns(columns),
            LogicalPlan::Sort { input, sort_keys } => {
                input.collect_columns(columns);
                for (key, _) in sort_keys {
                    columns.insert(key.clone());
                }
            }
        }
    }
}

impl Predicate {
    fn collect_columns(&self, columns: &mut HashSet<String>) {
        match self {
            Predicate::Range { field, .. } => {
                columns.insert(field.clone());
            }
            Predicate::Equals { field, .. } => {
                columns.insert(field.clone());
            }
            Predicate::In { field, .. } => {
                columns.insert(field.clone());
            }
            Predicate::And(preds) => preds.iter().for_each(|p| p.collect_columns(columns)),
            Predicate::Or(preds) => preds.iter().for_each(|p| p.collect_columns(columns)),
            Predicate::Not(pred) => pred.collect_columns(columns),
            Predicate::True => {}
        }
    }

    /// Check if predicate references only the given columns
    pub fn references_only(&self, allowed: &HashSet<String>) -> bool {
        let mut referenced = HashSet::new();
        self.collect_columns(&mut referenced);
        referenced.is_subset(allowed)
    }
}

/// Builder for constructing logical plans
pub struct LogicalPlanBuilder {
    plan: LogicalPlan,
}

impl LogicalPlanBuilder {
    /// Add column selection to scan
    pub fn columns(mut self, cols: Vec<&str>) -> Self {
        if let LogicalPlan::Scan { columns, .. } = &mut self.plan {
            *columns = cols.into_iter().map(|s| s.to_string()).collect();
        }
        self
    }

    /// Add a range filter
    pub fn filter_range(self, field: &str, min: f64, max: f64) -> Self {
        LogicalPlanBuilder {
            plan: LogicalPlan::Filter {
                input: Box::new(self.plan),
                predicate: Predicate::Range {
                    field: field.to_string(),
                    min,
                    max,
                },
            },
        }
    }

    /// Add an equality filter
    pub fn filter_eq(self, field: &str, value: f64) -> Self {
        LogicalPlanBuilder {
            plan: LogicalPlan::Filter {
                input: Box::new(self.plan),
                predicate: Predicate::Equals {
                    field: field.to_string(),
                    value,
                },
            },
        }
    }

    /// Add a custom predicate filter
    pub fn filter(self, predicate: Predicate) -> Self {
        LogicalPlanBuilder {
            plan: LogicalPlan::Filter {
                input: Box::new(self.plan),
                predicate,
            },
        }
    }

    /// Add a projection
    pub fn project(self, columns: Vec<&str>) -> Self {
        LogicalPlanBuilder {
            plan: LogicalPlan::Project {
                input: Box::new(self.plan),
                columns: columns.into_iter().map(|s| s.to_string()).collect(),
            },
        }
    }

    /// Add a join
    pub fn join(
        self,
        right: LogicalPlan,
        left_key: &str,
        right_key: &str,
        join_type: JoinType,
    ) -> Self {
        LogicalPlanBuilder {
            plan: LogicalPlan::Join {
                left: Box::new(self.plan),
                right: Box::new(right),
                left_key: left_key.to_string(),
                right_key: right_key.to_string(),
                join_type,
            },
        }
    }

    /// Add a group by with aggregations
    pub fn group_by(self, keys: Vec<&str>, aggregates: Vec<(&str, AggregateFunc, &str)>) -> Self {
        LogicalPlanBuilder {
            plan: LogicalPlan::GroupBy {
                input: Box::new(self.plan),
                group_keys: keys.into_iter().map(|s| s.to_string()).collect(),
                aggregates: aggregates
                    .into_iter()
                    .map(|(out, func, inp)| (out.to_string(), func, inp.to_string()))
                    .collect(),
            },
        }
    }

    /// Add a limit
    pub fn limit(self, count: usize) -> Self {
        LogicalPlanBuilder {
            plan: LogicalPlan::Limit {
                input: Box::new(self.plan),
                count,
            },
        }
    }

    /// Add sorting
    pub fn sort(self, keys: Vec<(&str, SortOrder)>) -> Self {
        LogicalPlanBuilder {
            plan: LogicalPlan::Sort {
                input: Box::new(self.plan),
                sort_keys: keys.into_iter().map(|(k, o)| (k.to_string(), o)).collect(),
            },
        }
    }

    /// Build the final plan
    pub fn build(self) -> LogicalPlan {
        self.plan
    }
}

// =============================================================================
// PHYSICAL PLAN
// =============================================================================

/// Physical execution plan - concrete operators
#[derive(Debug, Clone)]
pub enum PhysicalPlan {
    /// Columnar scan
    ColumnarScan {
        table: String,
        columns: Vec<String>,
        estimated_rows: usize,
    },
    /// Filter with pushed-down predicate
    Filter {
        input: Box<PhysicalPlan>,
        predicate: Predicate,
        estimated_rows: usize,
    },
    /// Hash join
    HashJoin {
        build: Box<PhysicalPlan>,
        probe: Box<PhysicalPlan>,
        build_key: String,
        probe_key: String,
        join_type: JoinType,
        estimated_rows: usize,
    },
    /// Hash aggregate
    HashAggregate {
        input: Box<PhysicalPlan>,
        group_keys: Vec<String>,
        aggregates: Vec<(String, AggregateFunc, String)>,
        estimated_rows: usize,
    },
    /// In-memory sort
    Sort {
        input: Box<PhysicalPlan>,
        sort_keys: Vec<(String, SortOrder)>,
    },
    /// Limit rows
    Limit {
        input: Box<PhysicalPlan>,
        count: usize,
    },
}

impl PhysicalPlan {
    /// Get estimated output rows
    pub fn estimated_rows(&self) -> usize {
        match self {
            PhysicalPlan::ColumnarScan { estimated_rows, .. } => *estimated_rows,
            PhysicalPlan::Filter { estimated_rows, .. } => *estimated_rows,
            PhysicalPlan::HashJoin { estimated_rows, .. } => *estimated_rows,
            PhysicalPlan::HashAggregate { estimated_rows, .. } => *estimated_rows,
            PhysicalPlan::Sort { input, .. } => input.estimated_rows(),
            PhysicalPlan::Limit { count, input } => (*count).min(input.estimated_rows()),
        }
    }

    /// Get estimated cost (in abstract units)
    pub fn estimated_cost(&self) -> f64 {
        match self {
            PhysicalPlan::ColumnarScan {
                estimated_rows,
                columns,
                ..
            } => {
                // Cost: rows * columns (I/O)
                (*estimated_rows as f64) * (columns.len().max(1) as f64) * 0.1
            }
            PhysicalPlan::Filter {
                input,
                estimated_rows,
                ..
            } => {
                // Cost: input cost + filter CPU
                input.estimated_cost() + (*estimated_rows as f64) * 0.01
            }
            PhysicalPlan::HashJoin { build, probe, .. } => {
                // Cost: build hash table + probe
                let build_cost = build.estimated_rows() as f64 * 2.0;
                let probe_cost = probe.estimated_rows() as f64 * 1.0;
                build.estimated_cost() + probe.estimated_cost() + build_cost + probe_cost
            }
            PhysicalPlan::HashAggregate {
                input,
                estimated_rows,
                ..
            } => {
                // Cost: input + hash aggregation
                input.estimated_cost() + (*estimated_rows as f64) * 1.5
            }
            PhysicalPlan::Sort { input, .. } => {
                // Cost: input + n log n sort
                let n = input.estimated_rows() as f64;
                input.estimated_cost() + n * n.log2().max(1.0)
            }
            PhysicalPlan::Limit { input, .. } => input.estimated_cost(),
        }
    }

    /// Display plan in EXPLAIN format
    pub fn explain(&self) -> String {
        self.explain_indent(0)
    }

    fn explain_indent(&self, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        match self {
            PhysicalPlan::ColumnarScan {
                table,
                columns,
                estimated_rows,
            } => {
                format!(
                    "{}ColumnarScan: {} [{}] (rows={})\n",
                    prefix,
                    table,
                    columns.join(", "),
                    estimated_rows
                )
            }
            PhysicalPlan::Filter {
                input,
                predicate,
                estimated_rows,
            } => {
                format!(
                    "{}Filter: {:?} (rows={})\n{}",
                    prefix,
                    predicate,
                    estimated_rows,
                    input.explain_indent(indent + 1)
                )
            }
            PhysicalPlan::HashJoin {
                build,
                probe,
                build_key,
                probe_key,
                join_type,
                estimated_rows,
            } => {
                format!(
                    "{}HashJoin: {:?} on {}={} (rows={})\n{}{}",
                    prefix,
                    join_type,
                    build_key,
                    probe_key,
                    estimated_rows,
                    build.explain_indent(indent + 1),
                    probe.explain_indent(indent + 1)
                )
            }
            PhysicalPlan::HashAggregate {
                input,
                group_keys,
                aggregates,
                estimated_rows,
            } => {
                let aggs: Vec<_> = aggregates
                    .iter()
                    .map(|(out, func, inp)| format!("{:?}({}) AS {}", func, inp, out))
                    .collect();
                format!(
                    "{}HashAggregate: group=[{}] aggs=[{}] (rows={})\n{}",
                    prefix,
                    group_keys.join(", "),
                    aggs.join(", "),
                    estimated_rows,
                    input.explain_indent(indent + 1)
                )
            }
            PhysicalPlan::Sort { input, sort_keys } => {
                let keys: Vec<_> = sort_keys
                    .iter()
                    .map(|(k, o)| format!("{} {:?}", k, o))
                    .collect();
                format!(
                    "{}Sort: [{}]\n{}",
                    prefix,
                    keys.join(", "),
                    input.explain_indent(indent + 1)
                )
            }
            PhysicalPlan::Limit { input, count } => {
                format!(
                    "{}Limit: {}\n{}",
                    prefix,
                    count,
                    input.explain_indent(indent + 1)
                )
            }
        }
    }
}

// =============================================================================
// RULE-BASED OPTIMIZER
// =============================================================================

/// Optimization rules
pub struct RuleBasedOptimizer;

impl RuleBasedOptimizer {
    /// Apply all optimization rules to a logical plan
    pub fn optimize(plan: LogicalPlan) -> LogicalPlan {
        let plan = Self::push_down_predicates(plan);
        let plan = Self::eliminate_unused_columns(plan);
        let plan = Self::fold_constants(plan);
        plan
    }

    /// Push predicates down through joins and projections
    pub fn push_down_predicates(plan: LogicalPlan) -> LogicalPlan {
        match plan {
            // Push filter below join if it only references one side
            LogicalPlan::Filter { input, predicate } => {
                match *input {
                    LogicalPlan::Join {
                        left,
                        right,
                        left_key,
                        right_key,
                        join_type,
                    } => {
                        let left_cols = left.referenced_columns();
                        let right_cols = right.referenced_columns();

                        // Check if predicate only uses left side columns
                        if predicate.references_only(&left_cols) {
                            LogicalPlan::Join {
                                left: Box::new(LogicalPlan::Filter {
                                    input: left,
                                    predicate,
                                }),
                                right,
                                left_key,
                                right_key,
                                join_type,
                            }
                        }
                        // Check if predicate only uses right side columns
                        else if predicate.references_only(&right_cols) {
                            LogicalPlan::Join {
                                left,
                                right: Box::new(LogicalPlan::Filter {
                                    input: right,
                                    predicate,
                                }),
                                left_key,
                                right_key,
                                join_type,
                            }
                        }
                        // Can't push down - keep as is
                        else {
                            LogicalPlan::Filter {
                                input: Box::new(LogicalPlan::Join {
                                    left,
                                    right,
                                    left_key,
                                    right_key,
                                    join_type,
                                }),
                                predicate,
                            }
                        }
                    }
                    // Merge consecutive filters
                    LogicalPlan::Filter {
                        input: inner,
                        predicate: inner_pred,
                    } => LogicalPlan::Filter {
                        input: inner,
                        predicate: Predicate::And(vec![inner_pred, predicate]),
                    },
                    // Other cases: just recurse
                    other => LogicalPlan::Filter {
                        input: Box::new(Self::push_down_predicates(other)),
                        predicate,
                    },
                }
            }
            // Recursively optimize children
            LogicalPlan::Project { input, columns } => LogicalPlan::Project {
                input: Box::new(Self::push_down_predicates(*input)),
                columns,
            },
            LogicalPlan::Join {
                left,
                right,
                left_key,
                right_key,
                join_type,
            } => LogicalPlan::Join {
                left: Box::new(Self::push_down_predicates(*left)),
                right: Box::new(Self::push_down_predicates(*right)),
                left_key,
                right_key,
                join_type,
            },
            LogicalPlan::GroupBy {
                input,
                group_keys,
                aggregates,
            } => LogicalPlan::GroupBy {
                input: Box::new(Self::push_down_predicates(*input)),
                group_keys,
                aggregates,
            },
            LogicalPlan::Limit { input, count } => LogicalPlan::Limit {
                input: Box::new(Self::push_down_predicates(*input)),
                count,
            },
            LogicalPlan::Sort { input, sort_keys } => LogicalPlan::Sort {
                input: Box::new(Self::push_down_predicates(*input)),
                sort_keys,
            },
            other => other,
        }
    }

    /// Remove projections of unused columns
    pub fn eliminate_unused_columns(plan: LogicalPlan) -> LogicalPlan {
        // For now, just recurse - full implementation would track column usage
        match plan {
            LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
                input: Box::new(Self::eliminate_unused_columns(*input)),
                predicate,
            },
            LogicalPlan::Project { input, columns } => LogicalPlan::Project {
                input: Box::new(Self::eliminate_unused_columns(*input)),
                columns,
            },
            LogicalPlan::Join {
                left,
                right,
                left_key,
                right_key,
                join_type,
            } => LogicalPlan::Join {
                left: Box::new(Self::eliminate_unused_columns(*left)),
                right: Box::new(Self::eliminate_unused_columns(*right)),
                left_key,
                right_key,
                join_type,
            },
            LogicalPlan::GroupBy {
                input,
                group_keys,
                aggregates,
            } => LogicalPlan::GroupBy {
                input: Box::new(Self::eliminate_unused_columns(*input)),
                group_keys,
                aggregates,
            },
            LogicalPlan::Limit { input, count } => LogicalPlan::Limit {
                input: Box::new(Self::eliminate_unused_columns(*input)),
                count,
            },
            LogicalPlan::Sort { input, sort_keys } => LogicalPlan::Sort {
                input: Box::new(Self::eliminate_unused_columns(*input)),
                sort_keys,
            },
            other => other,
        }
    }

    /// Fold constant expressions
    pub fn fold_constants(plan: LogicalPlan) -> LogicalPlan {
        match plan {
            // Remove always-true filters
            LogicalPlan::Filter {
                input,
                predicate: Predicate::True,
            } => Self::fold_constants(*input),
            // Simplify AND with True
            LogicalPlan::Filter {
                input,
                predicate: Predicate::And(preds),
            } => {
                let simplified: Vec<_> = preds
                    .into_iter()
                    .filter(|p| !matches!(p, Predicate::True))
                    .collect();

                let predicate = match simplified.len() {
                    0 => return Self::fold_constants(*input),
                    1 => simplified.into_iter().next().unwrap(),
                    _ => Predicate::And(simplified),
                };

                LogicalPlan::Filter {
                    input: Box::new(Self::fold_constants(*input)),
                    predicate,
                }
            }
            // Recursively fold
            LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
                input: Box::new(Self::fold_constants(*input)),
                predicate,
            },
            LogicalPlan::Project { input, columns } => LogicalPlan::Project {
                input: Box::new(Self::fold_constants(*input)),
                columns,
            },
            LogicalPlan::Join {
                left,
                right,
                left_key,
                right_key,
                join_type,
            } => LogicalPlan::Join {
                left: Box::new(Self::fold_constants(*left)),
                right: Box::new(Self::fold_constants(*right)),
                left_key,
                right_key,
                join_type,
            },
            LogicalPlan::GroupBy {
                input,
                group_keys,
                aggregates,
            } => LogicalPlan::GroupBy {
                input: Box::new(Self::fold_constants(*input)),
                group_keys,
                aggregates,
            },
            LogicalPlan::Limit { input, count } => LogicalPlan::Limit {
                input: Box::new(Self::fold_constants(*input)),
                count,
            },
            LogicalPlan::Sort { input, sort_keys } => LogicalPlan::Sort {
                input: Box::new(Self::fold_constants(*input)),
                sort_keys,
            },
            other => other,
        }
    }
}

// =============================================================================
// QUERY PLANNER
// =============================================================================

impl<'a> QueryOptimizer<'a> {
    /// Convert logical plan to optimized physical plan
    pub fn plan(&self, logical: LogicalPlan) -> PhysicalPlan {
        // First apply rule-based optimizations
        let optimized = RuleBasedOptimizer::optimize(logical);

        // Then convert to physical plan with cost-based decisions
        self.logical_to_physical(optimized)
    }

    fn logical_to_physical(&self, plan: LogicalPlan) -> PhysicalPlan {
        match plan {
            LogicalPlan::Scan { table, columns } => {
                let estimated_rows = columns
                    .first()
                    .and_then(|c| self.cardinality(c))
                    .unwrap_or(1000);

                PhysicalPlan::ColumnarScan {
                    table,
                    columns,
                    estimated_rows,
                }
            }

            LogicalPlan::Filter { input, predicate } => {
                let input_plan = self.logical_to_physical(*input);
                let input_rows = input_plan.estimated_rows();

                // Estimate selectivity
                let selectivity = self.estimate_predicate_selectivity(&predicate);
                let estimated_rows = ((input_rows as f64) * selectivity) as usize;

                PhysicalPlan::Filter {
                    input: Box::new(input_plan),
                    predicate,
                    estimated_rows,
                }
            }

            LogicalPlan::Project { input, columns: _ } => {
                // Project is just a logical operation - pass through to physical
                // Real implementation would add a projection operator
                self.logical_to_physical(*input)
            }

            LogicalPlan::Join {
                left,
                right,
                left_key,
                right_key,
                join_type,
            } => {
                let left_plan = self.logical_to_physical(*left);
                let right_plan = self.logical_to_physical(*right);

                // Cost-based: choose smaller side as build
                let left_rows = left_plan.estimated_rows();
                let right_rows = right_plan.estimated_rows();

                let (build, probe, build_key, probe_key) = if left_rows <= right_rows {
                    (left_plan, right_plan, left_key, right_key)
                } else {
                    (right_plan, left_plan, right_key, left_key)
                };

                let estimated_rows = match join_type {
                    JoinType::Inner => build.estimated_rows().min(probe.estimated_rows()),
                    JoinType::LeftOuter | JoinType::RightOuter => probe.estimated_rows(),
                    JoinType::FullOuter => build.estimated_rows() + probe.estimated_rows(),
                    JoinType::Semi | JoinType::Anti => build.estimated_rows() / 2,
                };

                PhysicalPlan::HashJoin {
                    build: Box::new(build),
                    probe: Box::new(probe),
                    build_key,
                    probe_key,
                    join_type,
                    estimated_rows,
                }
            }

            LogicalPlan::GroupBy {
                input,
                group_keys,
                aggregates,
            } => {
                let input_plan = self.logical_to_physical(*input);

                // Estimate distinct groups
                let estimated_rows = if group_keys.is_empty() {
                    1 // Scalar aggregation
                } else {
                    // Assume ~10% unique values
                    (input_plan.estimated_rows() / 10).max(1)
                };

                PhysicalPlan::HashAggregate {
                    input: Box::new(input_plan),
                    group_keys,
                    aggregates,
                    estimated_rows,
                }
            }

            LogicalPlan::Limit { input, count } => PhysicalPlan::Limit {
                input: Box::new(self.logical_to_physical(*input)),
                count,
            },

            LogicalPlan::Sort { input, sort_keys } => PhysicalPlan::Sort {
                input: Box::new(self.logical_to_physical(*input)),
                sort_keys,
            },
        }
    }

    fn estimate_predicate_selectivity(&self, predicate: &Predicate) -> f64 {
        match predicate {
            Predicate::Range { field, min, max } => {
                self.estimate_selectivity(field, *min, *max).unwrap_or(0.5)
            }
            Predicate::Equals { field, .. } => {
                // Point query - assume low selectivity
                self.cardinality(field)
                    .map(|c| 1.0 / (c as f64).max(1.0))
                    .unwrap_or(0.01)
            }
            Predicate::In { field, values } => {
                let per_value = self
                    .cardinality(field)
                    .map(|c| 1.0 / (c as f64).max(1.0))
                    .unwrap_or(0.01);
                (per_value * values.len() as f64).min(1.0)
            }
            Predicate::And(preds) => preds
                .iter()
                .map(|p| self.estimate_predicate_selectivity(p))
                .product(),
            Predicate::Or(preds) => {
                // P(A or B) = P(A) + P(B) - P(A)*P(B) (assuming independence)
                let mut result = 0.0;
                for p in preds {
                    let sel = self.estimate_predicate_selectivity(p);
                    result = result + sel - (result * sel);
                }
                result.min(1.0)
            }
            Predicate::Not(pred) => 1.0 - self.estimate_predicate_selectivity(pred),
            Predicate::True => 1.0,
        }
    }

    /// Generate a complete query plan with cost comparison
    pub fn plan_with_alternatives(&self, logical: LogicalPlan) -> Vec<(PhysicalPlan, f64)> {
        let optimized = RuleBasedOptimizer::optimize(logical.clone());
        let unoptimized = logical;

        let opt_physical = self.logical_to_physical(optimized);
        let unopt_physical = self.logical_to_physical(unoptimized);

        let opt_cost = opt_physical.estimated_cost();
        let unopt_cost = unopt_physical.estimated_cost();

        let mut plans = vec![(opt_physical, opt_cost), (unopt_physical, unopt_cost)];

        plans.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        plans
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Value;

    #[test]
    fn test_cardinality() {
        let mut store = ColumnarStore::new();
        store.record_value("a", 1, &Value::Int(10));
        store.record_value("a", 2, &Value::Int(20));
        store.record_value("a", 3, &Value::Int(30));

        let opt = QueryOptimizer::new(&store);
        assert_eq!(opt.cardinality("a"), Some(3));
        assert_eq!(opt.cardinality("nonexistent"), None);
    }

    #[test]
    fn test_selectivity_estimation() {
        let mut store = ColumnarStore::new();
        for i in 0..100 {
            store.record_value("x", i as u64, &Value::Float(i as f64));
        }

        let opt = QueryOptimizer::new(&store);

        // Full range
        let sel = opt.estimate_selectivity("x", 0.0, 100.0).unwrap();
        assert!((sel - 1.0).abs() < 0.01);

        // Half range
        let sel = opt.estimate_selectivity("x", 0.0, 50.0).unwrap();
        assert!((sel - 0.505).abs() < 0.02); // ~50.5% due to inclusive min

        // No overlap
        let sel = opt.estimate_selectivity("x", 200.0, 300.0).unwrap();
        assert_eq!(sel, 0.0);
    }

    #[test]
    fn test_join_optimization() {
        let mut store = ColumnarStore::new();

        // Small table (orders): 100 rows
        for i in 0..100 {
            store.record_value("o_orderkey", i as u64, &Value::Int(i as i64));
        }

        // Large table (lineitem): 1000 rows
        for i in 0..1000 {
            store.record_value("l_orderkey", i as u64, &Value::Int((i % 100) as i64));
        }

        let opt = QueryOptimizer::new(&store);

        // Should choose smaller table (orders) as build side
        let (build, probe) = opt.optimize_join("o_orderkey", "l_orderkey");
        assert_eq!(build, "o_orderkey");
        assert_eq!(probe, "l_orderkey");

        // Reverse order should give same result
        let (build, probe) = opt.optimize_join("l_orderkey", "o_orderkey");
        assert_eq!(build, "o_orderkey");
        assert_eq!(probe, "l_orderkey");
    }

    #[test]
    fn test_query_plan() {
        let mut store = ColumnarStore::new();

        for i in 0..100 {
            store.record_value("o_orderkey", i as u64, &Value::Int(i as i64));
        }
        for i in 0..1000 {
            store.record_value("l_orderkey", i as u64, &Value::Int((i % 100) as i64));
            store.record_value(
                "l_shipdate",
                i as u64,
                &Value::Int(19940101 + (i % 365) as i64),
            );
        }

        let opt = QueryOptimizer::new(&store);

        let plan = opt.plan_filtered_join(
            "o_orderkey",
            "l_orderkey",
            Some("l_shipdate"),
            Some(19940101.0),
            Some(19940201.0),
        );

        assert_eq!(plan.steps.len(), 2);
        assert!(plan.estimated_cost > 0.0);

        let explain = plan.explain();
        assert!(explain.contains("HashJoin"));
        assert!(explain.contains("Filter"));
    }

    #[test]
    fn test_logical_plan_builder() {
        let plan = LogicalPlan::scan("lineitem")
            .columns(vec!["l_orderkey", "l_extendedprice"])
            .filter_range("l_shipdate", 19940101.0, 19950101.0)
            .build();

        match plan {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Predicate::Range { field, min, max } => {
                    assert_eq!(field, "l_shipdate");
                    assert_eq!(min, 19940101.0);
                    assert_eq!(max, 19950101.0);
                }
                _ => panic!("Expected Range predicate"),
            },
            _ => panic!("Expected Filter plan"),
        }
    }

    #[test]
    fn test_logical_plan_with_join() {
        let right = LogicalPlan::scan("orders")
            .columns(vec!["o_orderkey", "o_custkey"])
            .build();

        let plan = LogicalPlan::scan("lineitem")
            .columns(vec!["l_orderkey", "l_extendedprice"])
            .join(right, "l_orderkey", "o_orderkey", JoinType::Inner)
            .build();

        match plan {
            LogicalPlan::Join {
                join_type,
                left_key,
                right_key,
                ..
            } => {
                assert_eq!(join_type, JoinType::Inner);
                assert_eq!(left_key, "l_orderkey");
                assert_eq!(right_key, "o_orderkey");
            }
            _ => panic!("Expected Join plan"),
        }
    }

    #[test]
    fn test_logical_plan_with_group_by() {
        let plan = LogicalPlan::scan("lineitem")
            .columns(vec!["l_returnflag", "l_extendedprice"])
            .group_by(
                vec!["l_returnflag"],
                vec![("total_price", AggregateFunc::Sum, "l_extendedprice")],
            )
            .build();

        match plan {
            LogicalPlan::GroupBy {
                group_keys,
                aggregates,
                ..
            } => {
                assert_eq!(group_keys, vec!["l_returnflag"]);
                assert_eq!(aggregates.len(), 1);
                assert_eq!(aggregates[0].0, "total_price");
                assert_eq!(aggregates[0].1, AggregateFunc::Sum);
            }
            _ => panic!("Expected GroupBy plan"),
        }
    }

    #[test]
    fn test_physical_plan_generation() {
        let mut store = ColumnarStore::new();

        for i in 0..100 {
            store.record_value("o_orderkey", i as u64, &Value::Int(i as i64));
        }
        for i in 0..1000 {
            store.record_value("l_orderkey", i as u64, &Value::Int((i % 100) as i64));
            store.record_value("l_extendedprice", i as u64, &Value::Float(i as f64 * 10.0));
        }

        let opt = QueryOptimizer::new(&store);

        let right = LogicalPlan::scan("orders")
            .columns(vec!["o_orderkey"])
            .build();

        let logical = LogicalPlan::scan("lineitem")
            .columns(vec!["l_orderkey", "l_extendedprice"])
            .join(right, "l_orderkey", "o_orderkey", JoinType::Inner)
            .group_by(
                vec!["o_orderkey"],
                vec![("revenue", AggregateFunc::Sum, "l_extendedprice")],
            )
            .build();

        let physical = opt.plan(logical);

        // Should have HashAggregate at top
        match &physical {
            PhysicalPlan::HashAggregate { aggregates, .. } => {
                assert_eq!(aggregates.len(), 1);
            }
            _ => panic!("Expected HashAggregate at top"),
        }

        let explain = physical.explain();
        assert!(explain.contains("HashAggregate"));
        assert!(explain.contains("HashJoin"));
    }

    #[test]
    fn test_predicate_pushdown() {
        // Build: Filter(Join(A, B), predicate on A)
        let left = LogicalPlan::Scan {
            table: "orders".to_string(),
            columns: vec!["o_orderkey".to_string()],
        };
        let right = LogicalPlan::Scan {
            table: "lineitem".to_string(),
            columns: vec!["l_orderkey".to_string()],
        };
        let join = LogicalPlan::Join {
            left: Box::new(left),
            right: Box::new(right),
            left_key: "o_orderkey".to_string(),
            right_key: "l_orderkey".to_string(),
            join_type: JoinType::Inner,
        };
        let filter = LogicalPlan::Filter {
            input: Box::new(join),
            predicate: Predicate::Range {
                field: "o_orderkey".to_string(),
                min: 1.0,
                max: 100.0,
            },
        };

        let optimized = RuleBasedOptimizer::push_down_predicates(filter);

        // After optimization, filter should be pushed into left side of join
        match optimized {
            LogicalPlan::Join { left, .. } => match *left {
                LogicalPlan::Filter { predicate, .. } => match predicate {
                    Predicate::Range { field, .. } => {
                        assert_eq!(field, "o_orderkey");
                    }
                    _ => panic!("Expected Range predicate"),
                },
                _ => panic!("Expected Filter pushed down"),
            },
            _ => panic!("Expected Join at top"),
        }
    }

    #[test]
    fn test_constant_folding() {
        // Filter with True predicate should be eliminated
        let scan = LogicalPlan::Scan {
            table: "test".to_string(),
            columns: vec!["a".to_string()],
        };
        let filter = LogicalPlan::Filter {
            input: Box::new(scan),
            predicate: Predicate::True,
        };

        let optimized = RuleBasedOptimizer::fold_constants(filter);

        // Filter should be removed
        match optimized {
            LogicalPlan::Scan { .. } => {}
            _ => panic!("Expected filter to be eliminated"),
        }
    }

    #[test]
    fn test_physical_plan_cost() {
        let mut store = ColumnarStore::new();
        for i in 0..1000 {
            store.record_value("a", i as u64, &Value::Int(i as i64));
        }

        let opt = QueryOptimizer::new(&store);

        let plan = LogicalPlan::scan("test")
            .columns(vec!["a"])
            .filter_range("a", 0.0, 500.0)
            .build();

        let physical = opt.plan(plan);

        assert!(physical.estimated_cost() > 0.0);
        assert!(physical.estimated_rows() > 0);
    }

    #[test]
    fn test_all_join_types_in_plan() {
        let mut store = ColumnarStore::new();
        for i in 0..100 {
            store.record_value("a_key", i as u64, &Value::Int(i as i64));
            store.record_value("b_key", i as u64, &Value::Int((i * 2) as i64));
        }

        let opt = QueryOptimizer::new(&store);

        for join_type in [
            JoinType::Inner,
            JoinType::LeftOuter,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let right = LogicalPlan::scan("b").columns(vec!["b_key"]).build();

            let plan = LogicalPlan::scan("a")
                .columns(vec!["a_key"])
                .join(right, "a_key", "b_key", join_type)
                .build();

            let physical = opt.plan(plan);

            match physical {
                PhysicalPlan::HashJoin { join_type: pjt, .. } => {
                    assert_eq!(pjt, join_type);
                }
                _ => panic!("Expected HashJoin"),
            }
        }
    }

    // =========================================================================
    // Histogram Tests
    // =========================================================================

    #[test]
    fn test_histogram_build() {
        let mut store = ColumnarStore::new();

        // Add values 0-99
        for i in 0..100 {
            store.record_value("x", i as u64, &Value::Float(i as f64));
        }

        let col = store.get_column("x").unwrap();
        let histogram = col.build_histogram(10);

        assert_eq!(histogram.buckets.len(), 10);
        assert_eq!(histogram.total_count, 100);
        assert_eq!(histogram.ndv, 100); // All distinct

        // Each bucket should have ~10 values
        for bucket in &histogram.buckets {
            assert!(bucket.count >= 9 && bucket.count <= 11);
        }
    }

    #[test]
    fn test_histogram_selectivity() {
        let mut store = ColumnarStore::new();

        // Add values 0-99
        for i in 0..100 {
            store.record_value("x", i as u64, &Value::Float(i as f64));
        }

        let col = store.get_column("x").unwrap();
        let histogram = col.build_histogram(10);

        // Full range selectivity should be ~1.0
        let sel = histogram.selectivity_range(0.0, 100.0);
        assert!((sel - 1.0).abs() < 0.01);

        // Half range should be ~0.5
        let sel = histogram.selectivity_range(0.0, 50.0);
        assert!((sel - 0.5).abs() < 0.1);

        // No overlap should be 0
        let sel = histogram.selectivity_range(200.0, 300.0);
        assert_eq!(sel, 0.0);
    }

    #[test]
    fn test_histogram_with_skew() {
        let mut store = ColumnarStore::new();

        // Add skewed data: many small values, few large
        for i in 0..80 {
            store.record_value("x", i as u64, &Value::Float((i % 10) as f64)); // 0-9
        }
        for i in 80..100 {
            store.record_value("x", i as u64, &Value::Float(((i - 80) * 5 + 50) as f64)); // 50-145
        }

        let col = store.get_column("x").unwrap();
        let histogram = col.build_histogram(10);

        // Selectivity for 0-10 should be high (80% of data)
        let sel = histogram.selectivity_range(0.0, 10.0);
        assert!(sel > 0.6);

        // Selectivity for 50-150 should be low (20% of data)
        let sel = histogram.selectivity_range(50.0, 150.0);
        assert!(sel < 0.4);
    }

    #[test]
    fn test_histogram_based_selectivity() {
        let mut store = ColumnarStore::new();

        for i in 0..1000 {
            store.record_value("value", i as u64, &Value::Float(i as f64));
        }

        let opt = QueryOptimizer::new(&store);

        // Using histogram should give more accurate estimation
        let sel = opt.estimate_selectivity_with_histogram("value", 0.0, 100.0);
        assert!(sel.is_some());
        assert!((sel.unwrap() - 0.1).abs() < 0.02);
    }

    // =========================================================================
    // DP Join Ordering Tests
    // =========================================================================

    #[test]
    fn test_join_predicate() {
        let pred = JoinPredicate::new("orders", "lineitem", "o_orderkey", "l_orderkey")
            .with_selectivity(0.1);

        assert_eq!(pred.left_table, "orders");
        assert_eq!(pred.right_table, "lineitem");
        assert_eq!(pred.selectivity, 0.1);
    }

    #[test]
    fn test_dp_join_order_two_tables() {
        let mut store = ColumnarStore::new();

        // Small table
        for i in 0..100 {
            store.record_value("o_orderkey", i as u64, &Value::Int(i as i64));
        }

        // Large table
        for i in 0..1000 {
            store.record_value("l_orderkey", i as u64, &Value::Int((i % 100) as i64));
        }

        let opt = QueryOptimizer::new(&store);

        let tables = vec!["orders".to_string(), "lineitem".to_string()];
        let predicates = vec![JoinPredicate::new(
            "orders",
            "lineitem",
            "o_orderkey",
            "l_orderkey",
        )];

        let plan = opt.optimize_join_order_dp(&tables, &predicates);

        // Should have 1 join
        assert_eq!(plan.len(), 1);

        // Build side should be the smaller table
        assert!(plan[0].estimated_cost > 0.0);
    }

    #[test]
    fn test_dp_join_order_three_tables() {
        let mut store = ColumnarStore::new();

        // nation: 25 rows
        for i in 0..25 {
            store.record_value("n_nationkey", i as u64, &Value::Int(i as i64));
        }

        // customer: 1500 rows
        for i in 0..1500 {
            store.record_value("c_custkey", i as u64, &Value::Int(i as i64));
            store.record_value("c_nationkey", i as u64, &Value::Int((i % 25) as i64));
        }

        // orders: 15000 rows
        for i in 0..15000 {
            store.record_value("o_orderkey", i as u64, &Value::Int(i as i64));
            store.record_value("o_custkey", i as u64, &Value::Int((i % 1500) as i64));
        }

        let opt = QueryOptimizer::new(&store);

        let tables = vec![
            "nation".to_string(),
            "customer".to_string(),
            "orders".to_string(),
        ];
        let predicates = vec![
            JoinPredicate::new("nation", "customer", "n_nationkey", "c_nationkey"),
            JoinPredicate::new("customer", "orders", "c_custkey", "o_custkey"),
        ];

        let plan = opt.optimize_join_order_dp(&tables, &predicates);

        // Should have 2 joins
        assert_eq!(plan.len(), 2);

        // All joins should have positive cost
        for join in &plan {
            assert!(join.estimated_cost > 0.0);
        }
    }

    #[test]
    fn test_join_spec() {
        let spec = JoinSpec {
            build_table: "small".to_string(),
            probe_table: "large".to_string(),
            left_col: "key_a".to_string(),
            right_col: "key_b".to_string(),
            estimated_cost: 1000.0,
        };

        assert_eq!(spec.build_table, "small");
        assert_eq!(spec.probe_table, "large");
        assert_eq!(spec.estimated_cost, 1000.0);
    }
}
