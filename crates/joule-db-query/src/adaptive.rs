//! Adaptive Query Optimizer
//!
//! A comprehensive adaptive query optimizer that learns from query execution
//! to improve future query plans. Features include:
//!
//! - Query plan caching with statistics
//! - Adaptive cardinality estimation based on actual results
//! - Plan regression detection and fallback
//! - Index recommendation based on query patterns
//! - Cost model updates from actual execution times
//! - Multi-arm bandit for plan selection using UCB

use crate::ast::{Expression, Operator, Query, Value};
use crate::error::{QueryError, QueryResult};
use crate::planner::PlanNode;
use crate::sql::SetOperationType;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::RwLock;
use std::time::{Duration, Instant};

// ============================================================================
// Query Fingerprint
// ============================================================================

/// A fingerprint uniquely identifies a query template (ignoring literal values)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryFingerprint {
    /// Hash of the normalized query structure
    pub hash: u64,
    /// Human-readable description
    pub description: String,
    /// Tables involved
    pub tables: Vec<String>,
    /// Columns referenced
    pub columns: Vec<String>,
}

impl QueryFingerprint {
    /// Create a fingerprint from a query
    pub fn from_query(query: &Query) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash query type
        format!("{:?}", query.query_type).hash(&mut hasher);

        // Hash source table
        if let Some(ref source) = query.source {
            source.hash(&mut hasher);
        }

        // Hash column names (sorted for consistency)
        let mut cols = query.columns.clone();
        cols.sort();
        cols.hash(&mut hasher);

        // Hash filter structure (without literal values)
        if let Some(ref filter) = query.filter {
            Self::hash_expression_structure(filter, &mut hasher);
        }

        // Hash joins
        for join in &query.joins {
            format!("{:?}", join.join_type).hash(&mut hasher);
            join.table.hash(&mut hasher);
        }

        // Hash group by
        format!("{:?}", query.group_by).hash(&mut hasher);

        // Hash order by columns
        for ob in &query.order_by {
            format!("{:?}", ob.expr).hash(&mut hasher);
            ob.descending.hash(&mut hasher);
        }

        let tables = query.source.iter().cloned().collect();
        let columns = query.columns.clone();

        Self {
            hash: hasher.finish(),
            description: Self::build_description(query),
            tables,
            columns,
        }
    }

    fn hash_expression_structure<H: Hasher>(expr: &Expression, hasher: &mut H) {
        match expr {
            Expression::Literal(_) => "literal".hash(hasher),
            Expression::Column(name) => {
                "column".hash(hasher);
                name.hash(hasher);
            }
            Expression::QualifiedColumn { table, column } => {
                "qualified".hash(hasher);
                table.hash(hasher);
                column.hash(hasher);
            }
            Expression::Binary { left, op, right } => {
                "binary".hash(hasher);
                format!("{:?}", op).hash(hasher);
                Self::hash_expression_structure(left, hasher);
                Self::hash_expression_structure(right, hasher);
            }
            Expression::Unary { op, expr } => {
                "unary".hash(hasher);
                format!("{:?}", op).hash(hasher);
                Self::hash_expression_structure(expr, hasher);
            }
            Expression::Function { name, args } => {
                "function".hash(hasher);
                name.hash(hasher);
                for arg in args {
                    Self::hash_expression_structure(arg, hasher);
                }
            }
            Expression::In { expr, negated, .. } => {
                "in".hash(hasher);
                negated.hash(hasher);
                Self::hash_expression_structure(expr, hasher);
            }
            Expression::Between { expr, negated, .. } => {
                "between".hash(hasher);
                negated.hash(hasher);
                Self::hash_expression_structure(expr, hasher);
            }
            Expression::IsNull { expr, negated } => {
                "isnull".hash(hasher);
                negated.hash(hasher);
                Self::hash_expression_structure(expr, hasher);
            }
            Expression::Like { expr, negated, .. } => {
                "like".hash(hasher);
                negated.hash(hasher);
                Self::hash_expression_structure(expr, hasher);
            }
            _ => "other".hash(hasher),
        }
    }

    fn build_description(query: &Query) -> String {
        let mut desc = format!("{:?}", query.query_type);
        if let Some(ref source) = query.source {
            desc.push_str(&format!(" FROM {}", source));
        }
        if query.filter.is_some() {
            desc.push_str(" WHERE ...");
        }
        if !query.joins.is_empty() {
            desc.push_str(&format!(" ({} joins)", query.joins.len()));
        }
        desc
    }
}

// ============================================================================
// Plan Statistics
// ============================================================================

/// Statistics about a query plan's execution
#[derive(Debug, Clone)]
pub struct PlanStatistics {
    /// Number of times this plan was executed
    pub execution_count: u64,
    /// Total execution time across all runs
    pub total_execution_time: Duration,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Minimum execution time
    pub min_execution_time: Duration,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Standard deviation of execution time
    pub stddev_execution_time: f64,
    /// Total rows returned across all runs
    pub total_rows_returned: u64,
    /// Average rows returned
    pub avg_rows_returned: f64,
    /// Estimated cost vs actual cost ratio
    pub cost_accuracy_ratio: f64,
    /// Cardinality estimation accuracy
    pub cardinality_accuracy: f64,
    /// Last execution timestamp
    pub last_executed: Instant,
    /// History of recent execution times (for trend analysis)
    execution_time_history: VecDeque<Duration>,
}

impl PlanStatistics {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self {
            execution_count: 0,
            total_execution_time: Duration::ZERO,
            avg_execution_time: Duration::ZERO,
            min_execution_time: Duration::MAX,
            max_execution_time: Duration::ZERO,
            stddev_execution_time: 0.0,
            total_rows_returned: 0,
            avg_rows_returned: 0.0,
            cost_accuracy_ratio: 1.0,
            cardinality_accuracy: 1.0,
            last_executed: Instant::now(),
            execution_time_history: VecDeque::with_capacity(100),
        }
    }

    /// Record an execution
    pub fn record_execution(
        &mut self,
        execution_time: Duration,
        rows_returned: u64,
        estimated_cost: f64,
        estimated_rows: u64,
    ) {
        self.execution_count += 1;
        self.total_execution_time += execution_time;
        self.total_rows_returned += rows_returned;
        self.last_executed = Instant::now();

        // Update min/max
        if execution_time < self.min_execution_time {
            self.min_execution_time = execution_time;
        }
        if execution_time > self.max_execution_time {
            self.max_execution_time = execution_time;
        }

        // Update averages
        self.avg_execution_time = self.total_execution_time / self.execution_count as u32;
        self.avg_rows_returned = self.total_rows_returned as f64 / self.execution_count as f64;

        // Update accuracy metrics
        let actual_cost = execution_time.as_secs_f64() * 1000.0; // Convert to ms
        if estimated_cost > 0.0 {
            self.cost_accuracy_ratio = actual_cost / estimated_cost;
        }

        if estimated_rows > 0 {
            self.cardinality_accuracy = rows_returned as f64 / estimated_rows as f64;
        }

        // Update history
        if self.execution_time_history.len() >= 100 {
            self.execution_time_history.pop_front();
        }
        self.execution_time_history.push_back(execution_time);

        // Calculate standard deviation
        self.update_stddev();
    }

    fn update_stddev(&mut self) {
        if self.execution_count < 2 {
            self.stddev_execution_time = 0.0;
            return;
        }

        let mean = self.avg_execution_time.as_secs_f64();
        let variance: f64 = self
            .execution_time_history
            .iter()
            .map(|t| {
                let diff = t.as_secs_f64() - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.execution_time_history.len() as f64 - 1.0);

        self.stddev_execution_time = variance.sqrt();
    }

    /// Check if performance is degrading
    pub fn is_degrading(&self) -> bool {
        if self.execution_time_history.len() < 10 {
            return false;
        }

        // Compare recent executions to historical average
        let recent: Vec<_> = self.execution_time_history.iter().rev().take(5).collect();
        let recent_avg = recent.iter().map(|d| d.as_secs_f64()).sum::<f64>() / 5.0;

        // If recent average is 50% higher than overall average, consider degrading
        recent_avg > self.avg_execution_time.as_secs_f64() * 1.5
    }
}

impl Default for PlanStatistics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Cached Plan Entry
// ============================================================================

/// A cached query plan with its statistics
#[derive(Debug, Clone)]
pub struct CachedPlanEntry {
    /// The query fingerprint
    pub fingerprint: QueryFingerprint,
    /// The plan node
    pub plan: PlanNode,
    /// Plan statistics
    pub statistics: PlanStatistics,
    /// Alternative plans (for multi-arm bandit selection)
    pub alternatives: Vec<AlternativePlan>,
    /// Creation time
    pub created_at: Instant,
    /// Last access time (for LRU eviction)
    pub last_accessed: Instant,
    /// Number of accesses (for frequency-based eviction)
    pub access_count: u64,
}

/// An alternative plan for the same query
#[derive(Debug, Clone)]
pub struct AlternativePlan {
    /// The alternative plan
    pub plan: PlanNode,
    /// Statistics for this alternative
    pub statistics: PlanStatistics,
    /// UCB score
    pub ucb_score: f64,
}

impl CachedPlanEntry {
    /// Create a new cached plan entry
    pub fn new(fingerprint: QueryFingerprint, plan: PlanNode) -> Self {
        let now = Instant::now();
        Self {
            fingerprint,
            plan,
            statistics: PlanStatistics::new(),
            alternatives: Vec::new(),
            created_at: now,
            last_accessed: now,
            access_count: 1,
        }
    }

    /// Mark as accessed
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Add an alternative plan
    pub fn add_alternative(&mut self, plan: PlanNode) {
        if self.alternatives.len() < 5 {
            self.alternatives.push(AlternativePlan {
                plan,
                statistics: PlanStatistics::new(),
                ucb_score: f64::MAX, // High initial score to encourage exploration
            });
        }
    }
}

// ============================================================================
// Plan Cache with LRU Eviction
// ============================================================================

/// Configuration for the plan cache
#[derive(Debug, Clone)]
pub struct PlanCacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Maximum age before eviction
    pub max_age: Duration,
    /// Enable LRU eviction
    pub lru_eviction: bool,
    /// Enable frequency-based eviction (combined with LRU)
    pub frequency_eviction: bool,
}

impl Default for PlanCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            max_age: Duration::from_secs(3600), // 1 hour
            lru_eviction: true,
            frequency_eviction: true,
        }
    }
}

/// LRU Plan Cache
pub struct PlanCache {
    /// Cached entries by fingerprint hash
    entries: HashMap<u64, CachedPlanEntry>,
    /// LRU order tracking
    lru_order: VecDeque<u64>,
    /// Configuration
    config: PlanCacheConfig,
    /// Cache statistics
    stats: CacheStatistics,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of evictions
    pub evictions: u64,
    /// Total entries added
    pub entries_added: u64,
}

impl CacheStatistics {
    /// Get hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl PlanCache {
    /// Create a new plan cache with default config
    pub fn new() -> Self {
        Self::with_config(PlanCacheConfig::default())
    }

    /// Create a new plan cache with custom config
    pub fn with_config(config: PlanCacheConfig) -> Self {
        Self {
            entries: HashMap::with_capacity(config.max_entries),
            lru_order: VecDeque::with_capacity(config.max_entries),
            config,
            stats: CacheStatistics::default(),
        }
    }

    /// Get a cached plan
    pub fn get(&mut self, fingerprint: &QueryFingerprint) -> Option<&mut CachedPlanEntry> {
        let hash = fingerprint.hash;

        if self.entries.contains_key(&hash) {
            self.stats.hits += 1;

            // Update LRU order
            self.lru_order.retain(|&h| h != hash);
            self.lru_order.push_back(hash);

            // Touch the entry
            if let Some(entry) = self.entries.get_mut(&hash) {
                entry.touch();
                return Some(entry);
            }
        }

        self.stats.misses += 1;
        None
    }

    /// Insert a plan into the cache
    pub fn insert(&mut self, entry: CachedPlanEntry) {
        let hash = entry.fingerprint.hash;

        // Evict if necessary
        while self.entries.len() >= self.config.max_entries {
            self.evict_one();
        }

        // Remove from LRU if already exists
        self.lru_order.retain(|&h| h != hash);

        // Insert
        self.entries.insert(hash, entry);
        self.lru_order.push_back(hash);
        self.stats.entries_added += 1;
    }

    /// Evict one entry based on policy
    fn evict_one(&mut self) {
        if self.config.lru_eviction && self.config.frequency_eviction {
            // Combined LRU + frequency eviction
            // Find the entry with lowest score = access_count / age
            let victim = self
                .entries
                .iter()
                .min_by(|(_, a), (_, b)| {
                    let age_a = a.created_at.elapsed().as_secs_f64().max(1.0);
                    let age_b = b.created_at.elapsed().as_secs_f64().max(1.0);
                    let score_a = a.access_count as f64 / age_a;
                    let score_b = b.access_count as f64 / age_b;
                    score_a
                        .partial_cmp(&score_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(k, _)| *k);

            if let Some(hash) = victim {
                self.entries.remove(&hash);
                self.lru_order.retain(|&h| h != hash);
                self.stats.evictions += 1;
            }
        } else if self.config.lru_eviction {
            // Pure LRU eviction
            if let Some(hash) = self.lru_order.pop_front() {
                self.entries.remove(&hash);
                self.stats.evictions += 1;
            }
        }
    }

    /// Remove stale entries
    pub fn cleanup_stale(&mut self) {
        let max_age = self.config.max_age;
        let stale: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, e)| e.last_accessed.elapsed() > max_age)
            .map(|(k, _)| *k)
            .collect();

        for hash in stale {
            self.entries.remove(&hash);
            self.lru_order.retain(|&h| h != hash);
            self.stats.evictions += 1;
        }
    }

    /// Get cache statistics
    pub fn statistics(&self) -> &CacheStatistics {
        &self.stats
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
    }
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Cardinality Estimator with Feedback Loop
// ============================================================================

/// Histogram bucket for column statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    /// Lower bound
    pub lower: Value,
    /// Upper bound  
    pub upper: Value,
    /// Number of distinct values
    pub ndv: u64,
    /// Frequency (row count)
    pub frequency: u64,
}

/// Column statistics
#[derive(Debug, Clone)]
pub struct ColumnStatistics {
    /// Column name
    pub column: String,
    /// Table name
    pub table: String,
    /// Number of distinct values
    pub ndv: u64,
    /// Null count
    pub null_count: u64,
    /// Minimum value
    pub min_value: Option<Value>,
    /// Maximum value
    pub max_value: Option<Value>,
    /// Histogram buckets
    pub histogram: Vec<HistogramBucket>,
    /// Most common values
    pub mcv: Vec<(Value, u64)>,
    /// Last updated
    pub last_updated: Instant,
}

impl ColumnStatistics {
    /// Create new column statistics
    pub fn new(table: &str, column: &str) -> Self {
        Self {
            column: column.to_string(),
            table: table.to_string(),
            ndv: 0,
            null_count: 0,
            min_value: None,
            max_value: None,
            histogram: Vec::new(),
            mcv: Vec::new(),
            last_updated: Instant::now(),
        }
    }
}

/// Table statistics
#[derive(Debug, Clone)]
pub struct TableStatistics {
    /// Table name
    pub table: String,
    /// Row count
    pub row_count: u64,
    /// Column statistics
    pub columns: HashMap<String, ColumnStatistics>,
    /// Last updated
    pub last_updated: Instant,
}

impl TableStatistics {
    /// Create new table statistics
    pub fn new(table: &str) -> Self {
        Self {
            table: table.to_string(),
            row_count: 1000, // Default estimate
            columns: HashMap::new(),
            last_updated: Instant::now(),
        }
    }
}

/// Cardinality estimator with feedback loop
pub struct CardinalityEstimator {
    /// Table statistics
    table_stats: HashMap<String, TableStatistics>,
    /// Historical estimation errors for learning
    estimation_errors: HashMap<String, VecDeque<f64>>,
    /// Learned correction factors per table
    correction_factors: HashMap<String, f64>,
    /// Default selectivity for unknown predicates
    default_selectivity: f64,
}

impl CardinalityEstimator {
    /// Create a new cardinality estimator
    pub fn new() -> Self {
        Self {
            table_stats: HashMap::new(),
            estimation_errors: HashMap::new(),
            correction_factors: HashMap::new(),
            default_selectivity: 0.1, // 10% default selectivity
        }
    }

    /// Estimate cardinality for a plan node
    pub fn estimate(&self, node: &PlanNode) -> u64 {
        let base_estimate = self.estimate_internal(node);

        // Apply learned correction factor
        let table = self.get_primary_table(node);
        let correction = self.correction_factors.get(&table).copied().unwrap_or(1.0);

        ((base_estimate as f64) * correction).max(1.0) as u64
    }

    fn estimate_internal(&self, node: &PlanNode) -> u64 {
        match node {
            PlanNode::Scan { table, filter, .. } => {
                let base = self
                    .table_stats
                    .get(table)
                    .map(|s| s.row_count)
                    .unwrap_or(1000);

                if let Some(expr) = filter {
                    let selectivity = self.estimate_selectivity(table, expr);
                    ((base as f64) * selectivity).max(1.0) as u64
                } else {
                    base
                }
            }
            PlanNode::IndexScan { table, filter, .. } => {
                let base = self
                    .table_stats
                    .get(table)
                    .map(|s| s.row_count)
                    .unwrap_or(1000);

                let selectivity = self.estimate_selectivity(table, filter);
                ((base as f64) * selectivity).max(1.0) as u64
            }
            PlanNode::Filter { input, predicate } => {
                let input_card = self.estimate_internal(input);
                let table = self.get_primary_table(input);
                let selectivity = self.estimate_selectivity(&table, predicate);
                ((input_card as f64) * selectivity).max(1.0) as u64
            }
            PlanNode::Project { input, .. } => self.estimate_internal(input),
            PlanNode::Sort { input, .. } => self.estimate_internal(input),
            PlanNode::Limit {
                input,
                limit,
                offset,
            } => {
                let input_card = self.estimate_internal(input);
                let start = offset.unwrap_or(0) as u64;
                let remaining = input_card.saturating_sub(start);
                limit.map(|l| remaining.min(l as u64)).unwrap_or(remaining)
            }
            PlanNode::Aggregate {
                input, group_by, ..
            } => {
                if group_by.is_empty() {
                    1
                } else {
                    let input_card = self.estimate_internal(input);
                    // Estimate unique groups as fraction of input
                    (input_card / 10).max(1)
                }
            }
            PlanNode::Join {
                left,
                right,
                join_type,
                ..
            } => {
                let left_card = self.estimate_internal(left);
                let right_card = self.estimate_internal(right);

                match join_type {
                    crate::ast::JoinType::Inner => {
                        // Assume 10% matching rate
                        ((left_card as f64) * (right_card as f64) * 0.1).max(1.0) as u64
                    }
                    crate::ast::JoinType::Left => left_card,
                    crate::ast::JoinType::Right => right_card,
                    crate::ast::JoinType::Full => left_card + right_card,
                    crate::ast::JoinType::Cross => left_card * right_card,
                }
            }
            PlanNode::Window { input, .. } => self.estimate_internal(input),
            PlanNode::Distinct { input } => {
                (self.estimate_internal(input) as f64 * 0.8).max(1.0) as u64
            }
            PlanNode::SetOp { left, right, .. } => {
                self.estimate_internal(left) + self.estimate_internal(right)
            }
            PlanNode::RecursiveCte { base, .. } => self.estimate_internal(base) * 10,
            PlanNode::VectorScan { k, .. } => *k as u64,
            PlanNode::WcojJoin { atoms, .. } => atoms.len() as u64 * 100,
            PlanNode::Empty => 0,
        }
    }

    fn get_primary_table(&self, node: &PlanNode) -> String {
        match node {
            PlanNode::Scan { table, .. } => table.clone(),
            PlanNode::IndexScan { table, .. } => table.clone(),
            PlanNode::Filter { input, .. } => self.get_primary_table(input),
            PlanNode::Project { input, .. } => self.get_primary_table(input),
            PlanNode::Sort { input, .. } => self.get_primary_table(input),
            PlanNode::Limit { input, .. } => self.get_primary_table(input),
            PlanNode::Aggregate { input, .. } => self.get_primary_table(input),
            PlanNode::Join { left, .. } => self.get_primary_table(left),
            PlanNode::Window { input, .. } => self.get_primary_table(input),
            PlanNode::Distinct { input } => self.get_primary_table(input),
            PlanNode::SetOp { left, .. } => self.get_primary_table(left),
            PlanNode::RecursiveCte { name, .. } => name.clone(),
            PlanNode::VectorScan { table, .. } => table.clone(),
            PlanNode::WcojJoin { atoms, .. } => {
                atoms.first().map(|(name, _)| name.clone()).unwrap_or_default()
            }
            PlanNode::Empty => String::new(),
        }
    }

    fn estimate_selectivity(&self, table: &str, expr: &Expression) -> f64 {
        match expr {
            Expression::Binary { left, op, right } => {
                match op {
                    Operator::And => {
                        let left_sel = self.estimate_selectivity(table, left);
                        let right_sel = self.estimate_selectivity(table, right);
                        left_sel * right_sel
                    }
                    Operator::Or => {
                        let left_sel = self.estimate_selectivity(table, left);
                        let right_sel = self.estimate_selectivity(table, right);
                        (left_sel + right_sel - left_sel * right_sel).min(1.0)
                    }
                    Operator::Eq => {
                        // Check for column = literal
                        if let (Expression::Column(col), Expression::Literal(_)) =
                            (left.as_ref(), right.as_ref())
                        {
                            self.selectivity_for_equality(table, col)
                        } else if let (Expression::Literal(_), Expression::Column(col)) =
                            (left.as_ref(), right.as_ref())
                        {
                            self.selectivity_for_equality(table, col)
                        } else {
                            self.default_selectivity
                        }
                    }
                    Operator::Lt | Operator::Le | Operator::Gt | Operator::Ge => {
                        0.3 // Range predicate default
                    }
                    Operator::Ne => {
                        0.9 // Most values are not equal
                    }
                    _ => self.default_selectivity,
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
                let base = (list.len() as f64 * 0.01).min(0.5);
                if *negated { 1.0 - base } else { base }
            }
            Expression::Between { negated, .. } => {
                let base = 0.25;
                if *negated { 1.0 - base } else { base }
            }
            Expression::Like {
                negated, pattern, ..
            } => {
                let base = if pattern.starts_with('%') { 0.25 } else { 0.1 };
                if *negated { 1.0 - base } else { base }
            }
            _ => self.default_selectivity,
        }
    }

    fn selectivity_for_equality(&self, table: &str, column: &str) -> f64 {
        if let Some(stats) = self.table_stats.get(table) {
            if let Some(col_stats) = stats.columns.get(column) {
                if col_stats.ndv > 0 {
                    return 1.0 / col_stats.ndv as f64;
                }
            }
        }
        self.default_selectivity
    }

    /// Update statistics based on actual execution results
    pub fn feedback(&mut self, table: &str, estimated: u64, actual: u64) {
        // Calculate estimation error
        let error = if estimated > 0 {
            actual as f64 / estimated as f64
        } else {
            1.0
        };

        // Update error history
        let history = self
            .estimation_errors
            .entry(table.to_string())
            .or_insert_with(|| VecDeque::with_capacity(100));

        if history.len() >= 100 {
            history.pop_front();
        }
        history.push_back(error);

        // Update correction factor using exponential moving average
        let avg_error: f64 = history.iter().sum::<f64>() / history.len() as f64;
        let current_factor = self.correction_factors.get(table).copied().unwrap_or(1.0);
        let alpha = 0.1; // Learning rate
        let new_factor = current_factor * (1.0 - alpha) + avg_error * alpha;

        self.correction_factors
            .insert(table.to_string(), new_factor);
    }

    /// Update table statistics
    pub fn update_table_stats(&mut self, table: &str, row_count: u64) {
        let stats = self
            .table_stats
            .entry(table.to_string())
            .or_insert_with(|| TableStatistics::new(table));

        stats.row_count = row_count;
        stats.last_updated = Instant::now();
    }

    /// Update column statistics
    pub fn update_column_stats(&mut self, table: &str, column: &str, ndv: u64) {
        let table_stats = self
            .table_stats
            .entry(table.to_string())
            .or_insert_with(|| TableStatistics::new(table));

        let col_stats = table_stats
            .columns
            .entry(column.to_string())
            .or_insert_with(|| ColumnStatistics::new(table, column));

        col_stats.ndv = ndv;
        col_stats.last_updated = Instant::now();
    }
}

impl Default for CardinalityEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Index Advisor
// ============================================================================

/// Index recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRecommendation {
    /// Table name
    pub table: String,
    /// Columns to index
    pub columns: Vec<String>,
    /// Index type recommendation
    pub index_type: IndexType,
    /// Estimated benefit (query speedup factor)
    pub estimated_benefit: f64,
    /// Number of queries that would benefit
    pub benefiting_queries: u64,
    /// Reason for recommendation
    pub reason: String,
}

/// Index type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// B-tree index (default, good for range queries)
    BTree,
    /// Hash index (good for equality queries)
    Hash,
    /// Composite index (multiple columns)
    Composite,
    /// Covering index (includes all queried columns)
    Covering,
}

/// Query pattern for index analysis
#[derive(Debug, Clone)]
struct QueryPattern {
    table: String,
    filter_columns: Vec<String>,
    join_columns: Vec<String>,
    order_columns: Vec<String>,
    frequency: u64,
    avg_execution_time: Duration,
}

/// Index advisor that analyzes query patterns
pub struct IndexAdvisor {
    /// Observed query patterns
    patterns: HashMap<String, QueryPattern>,
    /// Existing indexes
    existing_indexes: HashMap<String, Vec<Vec<String>>>,
    /// Minimum frequency to consider for recommendation
    min_frequency: u64,
    /// Minimum execution time to consider
    min_execution_time: Duration,
}

impl IndexAdvisor {
    /// Create a new index advisor
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            existing_indexes: HashMap::new(),
            min_frequency: 10,
            min_execution_time: Duration::from_millis(100),
        }
    }

    /// Record a query execution for pattern analysis
    pub fn record_query(&mut self, query: &Query, execution_time: Duration) {
        let Some(ref table) = query.source else {
            return;
        };

        let mut filter_columns = Vec::new();
        let mut join_columns = Vec::new();
        let mut order_columns = Vec::new();

        // Extract filter columns
        if let Some(ref filter) = query.filter {
            Self::extract_columns(filter, &mut filter_columns);
        }

        // Extract join columns
        for join in &query.joins {
            if let Some(ref cond) = join.condition {
                Self::extract_columns(cond, &mut join_columns);
            }
        }

        // Extract order columns
        for ob in &query.order_by {
            if let Expression::Column(name) = &ob.expr {
                order_columns.push(name.clone());
            } else {
                order_columns.push(format!("{:?}", ob.expr));
            }
        }

        let key = format!(
            "{}-{:?}-{:?}-{:?}",
            table, filter_columns, join_columns, order_columns
        );

        let pattern = self.patterns.entry(key).or_insert_with(|| QueryPattern {
            table: table.clone(),
            filter_columns: filter_columns.clone(),
            join_columns: join_columns.clone(),
            order_columns: order_columns.clone(),
            frequency: 0,
            avg_execution_time: Duration::ZERO,
        });

        pattern.frequency += 1;
        let total_time = pattern.avg_execution_time.as_nanos() as u128
            * (pattern.frequency - 1) as u128
            + execution_time.as_nanos() as u128;
        pattern.avg_execution_time =
            Duration::from_nanos((total_time / pattern.frequency as u128) as u64);
    }

    fn extract_columns(expr: &Expression, columns: &mut Vec<String>) {
        match expr {
            Expression::Column(name) => {
                if !columns.contains(name) {
                    columns.push(name.clone());
                }
            }
            Expression::QualifiedColumn { column, .. } => {
                if !columns.contains(column) {
                    columns.push(column.clone());
                }
            }
            Expression::Binary { left, right, .. } => {
                Self::extract_columns(left, columns);
                Self::extract_columns(right, columns);
            }
            Expression::Unary { expr, .. } => {
                Self::extract_columns(expr, columns);
            }
            Expression::In { expr, .. } => {
                Self::extract_columns(expr, columns);
            }
            Expression::Between { expr, .. } => {
                Self::extract_columns(expr, columns);
            }
            Expression::IsNull { expr, .. } => {
                Self::extract_columns(expr, columns);
            }
            Expression::Like { expr, .. } => {
                Self::extract_columns(expr, columns);
            }
            _ => {}
        }
    }

    /// Register an existing index
    pub fn register_index(&mut self, table: &str, columns: Vec<String>) {
        self.existing_indexes
            .entry(table.to_string())
            .or_default()
            .push(columns);
    }

    /// Generate index recommendations
    pub fn recommend(&self) -> Vec<IndexRecommendation> {
        let mut recommendations = Vec::new();

        for pattern in self.patterns.values() {
            // Skip low-frequency patterns
            if pattern.frequency < self.min_frequency {
                continue;
            }

            // Skip fast queries
            if pattern.avg_execution_time < self.min_execution_time {
                continue;
            }

            // Check filter columns
            for col in &pattern.filter_columns {
                if !self.has_index(&pattern.table, &[col.clone()]) {
                    recommendations.push(IndexRecommendation {
                        table: pattern.table.clone(),
                        columns: vec![col.clone()],
                        index_type: IndexType::BTree,
                        estimated_benefit: self.estimate_benefit(pattern),
                        benefiting_queries: pattern.frequency,
                        reason: format!("Column '{}' frequently used in WHERE clause", col),
                    });
                }
            }

            // Check join columns
            for col in &pattern.join_columns {
                if !self.has_index(&pattern.table, &[col.clone()]) {
                    recommendations.push(IndexRecommendation {
                        table: pattern.table.clone(),
                        columns: vec![col.clone()],
                        index_type: IndexType::Hash,
                        estimated_benefit: self.estimate_benefit(pattern) * 1.5,
                        benefiting_queries: pattern.frequency,
                        reason: format!("Column '{}' frequently used in JOIN condition", col),
                    });
                }
            }

            // Check composite index opportunity
            if pattern.filter_columns.len() >= 2 {
                let cols: Vec<_> = pattern.filter_columns.iter().take(3).cloned().collect();
                if !self.has_index(&pattern.table, &cols) {
                    recommendations.push(IndexRecommendation {
                        table: pattern.table.clone(),
                        columns: cols.clone(),
                        index_type: IndexType::Composite,
                        estimated_benefit: self.estimate_benefit(pattern) * 2.0,
                        benefiting_queries: pattern.frequency,
                        reason: format!(
                            "Columns {:?} frequently used together in WHERE clause",
                            cols
                        ),
                    });
                }
            }

            // Check covering index opportunity (filter + select columns)
            if !pattern.filter_columns.is_empty() && !pattern.order_columns.is_empty() {
                let mut cols = pattern.filter_columns.clone();
                cols.extend(pattern.order_columns.clone());
                cols.dedup();

                if cols.len() <= 5 && !self.has_index(&pattern.table, &cols) {
                    recommendations.push(IndexRecommendation {
                        table: pattern.table.clone(),
                        columns: cols.clone(),
                        index_type: IndexType::Covering,
                        estimated_benefit: self.estimate_benefit(pattern) * 3.0,
                        benefiting_queries: pattern.frequency,
                        reason: "Covering index would eliminate table lookup".to_string(),
                    });
                }
            }
        }

        // Sort by estimated benefit
        recommendations.sort_by(|a, b| {
            b.estimated_benefit
                .partial_cmp(&a.estimated_benefit)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate (keep highest benefit for same columns)
        let mut seen: HashMap<String, usize> = HashMap::new();
        recommendations.retain(|r| {
            let key = format!("{}-{:?}", r.table, r.columns);
            if seen.contains_key(&key) {
                false
            } else {
                seen.insert(key, seen.len());
                true
            }
        });

        recommendations
    }

    fn has_index(&self, table: &str, columns: &[String]) -> bool {
        if let Some(indexes) = self.existing_indexes.get(table) {
            for idx_cols in indexes {
                // Check if columns are a prefix of an existing index
                if idx_cols.len() >= columns.len()
                    && idx_cols.iter().zip(columns).all(|(a, b)| a == b)
                {
                    return true;
                }
            }
        }
        false
    }

    fn estimate_benefit(&self, pattern: &QueryPattern) -> f64 {
        // Simple heuristic: longer queries benefit more from indexing
        let time_factor = pattern.avg_execution_time.as_secs_f64() * 10.0;
        let freq_factor = (pattern.frequency as f64).ln().max(1.0);
        time_factor * freq_factor
    }
}

impl Default for IndexAdvisor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Cost Model with Learned Coefficients
// ============================================================================

/// Cost model coefficients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostCoefficients {
    /// Cost per sequential page read
    pub seq_page_cost: f64,
    /// Cost per random page read
    pub random_page_cost: f64,
    /// Cost per tuple processed
    pub cpu_tuple_cost: f64,
    /// Cost per index entry processed
    pub cpu_index_tuple_cost: f64,
    /// Cost per operator evaluation
    pub cpu_operator_cost: f64,
    /// Cost per comparison in sort
    pub sort_comparison_cost: f64,
    /// Cost per hash operation
    pub hash_cost: f64,
}

impl Default for CostCoefficients {
    fn default() -> Self {
        Self {
            seq_page_cost: 1.0,
            random_page_cost: 4.0,
            cpu_tuple_cost: 0.01,
            cpu_index_tuple_cost: 0.005,
            cpu_operator_cost: 0.0025,
            sort_comparison_cost: 0.0001,
            hash_cost: 0.00005,
        }
    }
}

/// Cost model with learned coefficients
pub struct CostModel {
    /// Current coefficients
    coefficients: CostCoefficients,
    /// Learning rate for coefficient updates
    learning_rate: f64,
    /// History of cost predictions vs actuals
    prediction_history: VecDeque<(f64, f64)>,
    /// Coefficient update history for trend analysis
    coefficient_history: VecDeque<CostCoefficients>,
}

impl CostModel {
    /// Create a new cost model with default coefficients
    pub fn new() -> Self {
        Self {
            coefficients: CostCoefficients::default(),
            learning_rate: 0.01,
            prediction_history: VecDeque::with_capacity(1000),
            coefficient_history: VecDeque::with_capacity(100),
        }
    }

    /// Estimate cost of a plan
    pub fn estimate(&self, node: &PlanNode, row_estimate: u64) -> f64 {
        self.estimate_node(node, row_estimate)
    }

    fn estimate_node(&self, node: &PlanNode, row_estimate: u64) -> f64 {
        match node {
            PlanNode::Scan { filter, .. } => {
                let scan_cost = (row_estimate as f64) * self.coefficients.cpu_tuple_cost;
                let filter_cost = if filter.is_some() {
                    (row_estimate as f64) * self.coefficients.cpu_operator_cost
                } else {
                    0.0
                };
                scan_cost + filter_cost
            }
            PlanNode::IndexScan { .. } => {
                // Index scan is cheaper than sequential scan
                // Assume it touches fewer pages and has index lookup overhead
                let index_overhead = self.coefficients.random_page_cost * 3.0; // Assume depth 3 tree
                let scan_cost = (row_estimate as f64) * self.coefficients.cpu_index_tuple_cost;
                index_overhead + scan_cost
            }
            PlanNode::Filter { input, .. } => {
                let input_cost = self.estimate_node(input, row_estimate);
                let filter_cost = (row_estimate as f64) * self.coefficients.cpu_operator_cost;
                input_cost + filter_cost
            }
            PlanNode::Project { input, columns } => {
                let input_cost = self.estimate_node(input, row_estimate);
                let project_cost = (row_estimate as f64)
                    * (columns.len() as f64)
                    * self.coefficients.cpu_tuple_cost;
                input_cost + project_cost
            }
            PlanNode::Sort { input, .. } => {
                let input_cost = self.estimate_node(input, row_estimate);
                let n = row_estimate as f64;
                let sort_cost = if n > 1.0 {
                    n * n.log2() * self.coefficients.sort_comparison_cost
                } else {
                    0.0
                };
                input_cost + sort_cost
            }
            PlanNode::Limit { input, .. } => self.estimate_node(input, row_estimate),
            PlanNode::Aggregate {
                input, aggregates, ..
            } => {
                let input_cost = self.estimate_node(input, row_estimate);
                let agg_cost = (row_estimate as f64)
                    * (aggregates.len().max(1) as f64)
                    * self.coefficients.cpu_operator_cost;
                input_cost + agg_cost
            }
            PlanNode::Join { left, right, .. } => {
                let left_cost = self.estimate_node(left, row_estimate);
                let right_rows = (row_estimate as f64 * 0.1) as u64;
                let right_cost = self.estimate_node(right, right_rows.max(1));
                let join_cost = (row_estimate as f64) * self.coefficients.hash_cost;
                left_cost + right_cost + join_cost
            }
            PlanNode::Window {
                input,
                window_functions,
            } => {
                let input_cost = self.estimate_node(input, row_estimate);
                let window_cost = (row_estimate as f64)
                    * (window_functions.len().max(1) as f64)
                    * self.coefficients.cpu_operator_cost
                    * 10.0; // Window functions are more expensive
                input_cost + window_cost
            }
            PlanNode::SetOp { left, right, op } => {
                let left_cost = self.estimate_node(left, row_estimate);
                let right_cost = self.estimate_node(right, row_estimate);
                let needs_dedup = !matches!(
                    op,
                    SetOperationType::UnionAll
                        | SetOperationType::ExceptAll
                        | SetOperationType::IntersectAll
                );
                let dedup_cost = if needs_dedup {
                    (row_estimate as f64) * self.coefficients.cpu_operator_cost * 2.0
                } else {
                    0.0
                };
                left_cost + right_cost + dedup_cost
            }
            PlanNode::Distinct { input } => {
                let input_cost = self.estimate_node(input, row_estimate);
                let dedup_cost = (row_estimate as f64) * self.coefficients.cpu_operator_cost * 2.0;
                input_cost + dedup_cost
            }
            PlanNode::RecursiveCte {
                base, recursive, ..
            } => {
                let base_cost = self.estimate_node(base, row_estimate);
                let recursive_cost = self.estimate_node(recursive, row_estimate);
                base_cost + recursive_cost * 10.0
            }
            PlanNode::VectorScan { k, .. } => {
                // HNSW-style: log2(N) * k, approximate with row_estimate as N
                let n = row_estimate as f64;
                if n > 0.0 {
                    n.log2() * (*k as f64)
                } else {
                    *k as f64
                }
            }
            PlanNode::WcojJoin { atoms, .. } => atoms.len() as f64 * 100.0,
            PlanNode::Empty => 0.0,
        }
    }

    /// Update cost model based on actual execution
    pub fn feedback(&mut self, predicted: f64, actual_time: Duration) {
        let actual = actual_time.as_secs_f64() * 1000.0; // Convert to ms scale

        // Record for history
        if self.prediction_history.len() >= 1000 {
            self.prediction_history.pop_front();
        }
        self.prediction_history.push_back((predicted, actual));

        // Calculate prediction error
        let error_ratio = if predicted > 0.0 {
            actual / predicted
        } else {
            1.0
        };

        // Update coefficients using gradient descent
        // If we underestimate (error_ratio > 1), increase costs
        // If we overestimate (error_ratio < 1), decrease costs
        let adjustment = (error_ratio - 1.0) * self.learning_rate;

        self.coefficients.cpu_tuple_cost *= 1.0 + adjustment;
        self.coefficients.cpu_operator_cost *= 1.0 + adjustment;
        self.coefficients.sort_comparison_cost *= 1.0 + adjustment;
        self.coefficients.hash_cost *= 1.0 + adjustment;

        // Clamp coefficients to reasonable ranges
        self.coefficients.cpu_tuple_cost = self.coefficients.cpu_tuple_cost.clamp(0.001, 1.0);
        self.coefficients.cpu_operator_cost =
            self.coefficients.cpu_operator_cost.clamp(0.0001, 0.1);
        self.coefficients.sort_comparison_cost =
            self.coefficients.sort_comparison_cost.clamp(0.00001, 0.01);
        self.coefficients.hash_cost = self.coefficients.hash_cost.clamp(0.000001, 0.001);

        // Save coefficient snapshot
        if self.coefficient_history.len() >= 100 {
            self.coefficient_history.pop_front();
        }
        self.coefficient_history
            .push_back(self.coefficients.clone());
    }

    /// Get current coefficients
    pub fn coefficients(&self) -> &CostCoefficients {
        &self.coefficients
    }

    /// Get prediction accuracy (R-squared)
    pub fn prediction_accuracy(&self) -> f64 {
        if self.prediction_history.len() < 10 {
            return 0.0;
        }

        let mean_actual: f64 = self.prediction_history.iter().map(|(_, a)| a).sum::<f64>()
            / self.prediction_history.len() as f64;

        let ss_tot: f64 = self
            .prediction_history
            .iter()
            .map(|(_, a)| (a - mean_actual).powi(2))
            .sum();

        let ss_res: f64 = self
            .prediction_history
            .iter()
            .map(|(p, a)| (a - p).powi(2))
            .sum();

        if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        }
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Plan Selector using UCB (Upper Confidence Bound)
// ============================================================================

/// Plan selection arm for multi-arm bandit
#[derive(Debug, Clone)]
pub struct PlanArm {
    /// Plan identifier
    pub plan_id: usize,
    /// Number of times selected
    pub selections: u64,
    /// Total reward (negative of execution time)
    pub total_reward: f64,
    /// Average reward
    pub avg_reward: f64,
    /// UCB score
    pub ucb_score: f64,
}

/// Multi-arm bandit plan selector using UCB1
pub struct PlanSelector {
    /// Exploration parameter (higher = more exploration)
    exploration_param: f64,
    /// Total selections across all arms
    total_selections: u64,
    /// Minimum selections before exploitation
    min_selections: u64,
}

impl PlanSelector {
    /// Create a new plan selector
    pub fn new() -> Self {
        Self {
            exploration_param: 2.0_f64.sqrt(),
            total_selections: 0,
            min_selections: 5,
        }
    }

    /// Create with custom exploration parameter
    pub fn with_exploration(exploration_param: f64) -> Self {
        Self {
            exploration_param,
            total_selections: 0,
            min_selections: 5,
        }
    }

    /// Select best plan using UCB1 algorithm
    pub fn select(&mut self, arms: &mut [PlanArm]) -> usize {
        if arms.is_empty() {
            return 0;
        }

        self.total_selections += 1;

        // First, ensure all arms have minimum selections
        for (idx, arm) in arms.iter().enumerate() {
            if arm.selections < self.min_selections {
                return idx;
            }
        }

        // Calculate UCB scores
        let ln_total = (self.total_selections as f64).ln();

        for arm in arms.iter_mut() {
            let exploitation = arm.avg_reward;
            let exploration = self.exploration_param * (ln_total / arm.selections as f64).sqrt();
            arm.ucb_score = exploitation + exploration;
        }

        // Select arm with highest UCB score
        arms.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.ucb_score
                    .partial_cmp(&b.ucb_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Update arm with reward
    pub fn update_arm(arm: &mut PlanArm, execution_time: Duration) {
        // Reward is negative of execution time (lower is better)
        let reward = -execution_time.as_secs_f64();

        arm.selections += 1;
        arm.total_reward += reward;
        arm.avg_reward = arm.total_reward / arm.selections as f64;
    }

    /// Create a new arm
    pub fn create_arm(plan_id: usize) -> PlanArm {
        PlanArm {
            plan_id,
            selections: 0,
            total_reward: 0.0,
            avg_reward: 0.0,
            ucb_score: f64::MAX, // High initial score for exploration
        }
    }

    /// Get exploration vs exploitation ratio
    pub fn exploration_ratio(&self, arms: &[PlanArm]) -> f64 {
        if arms.is_empty() || self.total_selections == 0 {
            return 1.0;
        }

        // Count how many times we selected non-best arms
        let best_avg = arms
            .iter()
            .map(|a| a.avg_reward)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let exploration_selections: u64 = arms
            .iter()
            .filter(|a| a.avg_reward < best_avg * 0.9)
            .map(|a| a.selections)
            .sum();

        exploration_selections as f64 / self.total_selections as f64
    }
}

impl Default for PlanSelector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Adaptive Optimizer
// ============================================================================

/// Adaptive optimizer configuration
#[derive(Debug, Clone)]
pub struct AdaptiveOptimizerConfig {
    /// Plan cache configuration
    pub cache_config: PlanCacheConfig,
    /// Enable plan regression detection
    pub regression_detection: bool,
    /// Regression threshold (multiplier over baseline)
    pub regression_threshold: f64,
    /// Enable index recommendations
    pub index_recommendations: bool,
    /// Enable cost model learning
    pub cost_model_learning: bool,
    /// Enable UCB plan selection
    pub ucb_selection: bool,
}

impl Default for AdaptiveOptimizerConfig {
    fn default() -> Self {
        Self {
            cache_config: PlanCacheConfig::default(),
            regression_detection: true,
            regression_threshold: 2.0,
            index_recommendations: true,
            cost_model_learning: true,
            ucb_selection: true,
        }
    }
}

/// The main adaptive query optimizer
pub struct AdaptiveOptimizer {
    /// Plan cache
    cache: RwLock<PlanCache>,
    /// Cardinality estimator
    cardinality_estimator: RwLock<CardinalityEstimator>,
    /// Index advisor
    index_advisor: RwLock<IndexAdvisor>,
    /// Cost model
    cost_model: RwLock<CostModel>,
    /// Plan selector
    plan_selector: RwLock<PlanSelector>,
    /// Configuration
    config: AdaptiveOptimizerConfig,
    /// Regression fallback plans
    fallback_plans: RwLock<HashMap<u64, PlanNode>>,
}

impl AdaptiveOptimizer {
    /// Create a new adaptive optimizer with default configuration
    pub fn new() -> Self {
        Self::with_config(AdaptiveOptimizerConfig::default())
    }

    /// Create a new adaptive optimizer with custom configuration
    pub fn with_config(config: AdaptiveOptimizerConfig) -> Self {
        Self {
            cache: RwLock::new(PlanCache::with_config(config.cache_config.clone())),
            cardinality_estimator: RwLock::new(CardinalityEstimator::new()),
            index_advisor: RwLock::new(IndexAdvisor::new()),
            cost_model: RwLock::new(CostModel::new()),
            plan_selector: RwLock::new(PlanSelector::new()),
            config,
            fallback_plans: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create an optimized plan for a query
    pub fn optimize(&self, query: &Query, plan: PlanNode) -> QueryResult<PlanNode> {
        let fingerprint = QueryFingerprint::from_query(query);

        // Try to get cached plan
        {
            let mut cache = self
                .cache
                .write()
                .map_err(|_| QueryError::ExecutionError("Cache lock poisoned".to_string()))?;

            if let Some(entry) = cache.get(&fingerprint) {
                // Check for regression
                if self.config.regression_detection && entry.statistics.is_degrading() {
                    // Try fallback plan
                    let fallbacks = self.fallback_plans.read().map_err(|_| {
                        QueryError::ExecutionError("Fallback lock poisoned".to_string())
                    })?;

                    if let Some(fallback) = fallbacks.get(&fingerprint.hash) {
                        return Ok(fallback.clone());
                    }
                }

                // Use UCB selection if enabled and alternatives exist
                if self.config.ucb_selection && !entry.alternatives.is_empty() {
                    let mut arms: Vec<PlanArm> = entry
                        .alternatives
                        .iter()
                        .enumerate()
                        .map(|(i, alt)| PlanArm {
                            plan_id: i,
                            selections: alt.statistics.execution_count,
                            total_reward: -(alt.statistics.avg_execution_time.as_secs_f64()),
                            avg_reward: if alt.statistics.execution_count > 0 {
                                -(alt.statistics.avg_execution_time.as_secs_f64())
                            } else {
                                0.0
                            },
                            ucb_score: alt.ucb_score,
                        })
                        .collect();

                    // Add main plan as arm 0
                    arms.insert(
                        0,
                        PlanArm {
                            plan_id: 0,
                            selections: entry.statistics.execution_count,
                            total_reward: -(entry.statistics.avg_execution_time.as_secs_f64()),
                            avg_reward: if entry.statistics.execution_count > 0 {
                                -(entry.statistics.avg_execution_time.as_secs_f64())
                            } else {
                                0.0
                            },
                            ucb_score: 0.0,
                        },
                    );

                    let mut selector = self.plan_selector.write().map_err(|_| {
                        QueryError::ExecutionError("Selector lock poisoned".to_string())
                    })?;

                    let selected = selector.select(&mut arms);

                    if selected == 0 {
                        return Ok(entry.plan.clone());
                    } else {
                        return Ok(entry.alternatives[selected - 1].plan.clone());
                    }
                }

                return Ok(entry.plan.clone());
            }
        }

        // Estimate cardinality
        let cardinality = {
            let estimator = self
                .cardinality_estimator
                .read()
                .map_err(|_| QueryError::ExecutionError("Estimator lock poisoned".to_string()))?;
            estimator.estimate(&plan)
        };

        // Estimate cost (kept for future use in plan comparison)
        let _cost = {
            let cost_model = self
                .cost_model
                .read()
                .map_err(|_| QueryError::ExecutionError("Cost model lock poisoned".to_string()))?;
            cost_model.estimate(&plan, cardinality)
        };

        // Cache the plan
        {
            let mut cache = self
                .cache
                .write()
                .map_err(|_| QueryError::ExecutionError("Cache lock poisoned".to_string()))?;

            let entry = CachedPlanEntry::new(fingerprint, plan.clone());
            cache.insert(entry);
        }

        Ok(plan)
    }

    /// Record execution feedback
    pub fn feedback(
        &self,
        query: &Query,
        execution_time: Duration,
        rows_returned: u64,
    ) -> QueryResult<()> {
        let fingerprint = QueryFingerprint::from_query(query);

        // Update cache statistics
        {
            let mut cache = self
                .cache
                .write()
                .map_err(|_| QueryError::ExecutionError("Cache lock poisoned".to_string()))?;

            if let Some(entry) = cache.get(&fingerprint) {
                let estimated_cost = entry.plan.estimate_cost();
                let estimated_rows = entry.plan.estimate_rows() as u64;

                entry.statistics.record_execution(
                    execution_time,
                    rows_returned,
                    estimated_cost,
                    estimated_rows,
                );

                // Check for regression and save baseline as fallback
                if self.config.regression_detection {
                    if entry.statistics.execution_count == 10 {
                        // Save baseline after 10 executions
                        let mut fallbacks = self.fallback_plans.write().map_err(|_| {
                            QueryError::ExecutionError("Fallback lock poisoned".to_string())
                        })?;
                        fallbacks.insert(fingerprint.hash, entry.plan.clone());
                    }
                }
            }
        }

        // Update cardinality estimator
        if let Some(ref table) = query.source {
            let mut estimator = self
                .cardinality_estimator
                .write()
                .map_err(|_| QueryError::ExecutionError("Estimator lock poisoned".to_string()))?;

            let estimated = {
                let cache = self
                    .cache
                    .read()
                    .map_err(|_| QueryError::ExecutionError("Cache lock poisoned".to_string()))?;
                cache
                    .entries
                    .get(&fingerprint.hash)
                    .map(|e| e.plan.estimate_rows() as u64)
                    .unwrap_or(1000)
            };

            estimator.feedback(table, estimated, rows_returned);
        }

        // Update cost model
        if self.config.cost_model_learning {
            let predicted = {
                let cache = self
                    .cache
                    .read()
                    .map_err(|_| QueryError::ExecutionError("Cache lock poisoned".to_string()))?;
                cache
                    .entries
                    .get(&fingerprint.hash)
                    .map(|e| e.plan.estimate_cost())
                    .unwrap_or(100.0)
            };

            let mut cost_model = self
                .cost_model
                .write()
                .map_err(|_| QueryError::ExecutionError("Cost model lock poisoned".to_string()))?;
            cost_model.feedback(predicted, execution_time);
        }

        // Update index advisor
        if self.config.index_recommendations {
            let mut advisor = self
                .index_advisor
                .write()
                .map_err(|_| QueryError::ExecutionError("Advisor lock poisoned".to_string()))?;
            advisor.record_query(query, execution_time);
        }

        Ok(())
    }

    /// Add an alternative plan for a query
    pub fn add_alternative_plan(&self, query: &Query, plan: PlanNode) -> QueryResult<()> {
        let fingerprint = QueryFingerprint::from_query(query);

        let mut cache = self
            .cache
            .write()
            .map_err(|_| QueryError::ExecutionError("Cache lock poisoned".to_string()))?;

        if let Some(entry) = cache.get(&fingerprint) {
            entry.add_alternative(plan);
        }

        Ok(())
    }

    /// Get index recommendations
    pub fn recommend_indexes(&self) -> QueryResult<Vec<IndexRecommendation>> {
        let advisor = self
            .index_advisor
            .read()
            .map_err(|_| QueryError::ExecutionError("Advisor lock poisoned".to_string()))?;
        Ok(advisor.recommend())
    }

    /// Get cache statistics
    pub fn cache_statistics(&self) -> QueryResult<CacheStatistics> {
        let cache = self
            .cache
            .read()
            .map_err(|_| QueryError::ExecutionError("Cache lock poisoned".to_string()))?;
        Ok(cache.statistics().clone())
    }

    /// Get cost model accuracy
    pub fn cost_model_accuracy(&self) -> QueryResult<f64> {
        let cost_model = self
            .cost_model
            .read()
            .map_err(|_| QueryError::ExecutionError("Cost model lock poisoned".to_string()))?;
        Ok(cost_model.prediction_accuracy())
    }

    /// Update table statistics
    pub fn update_table_stats(&self, table: &str, row_count: u64) -> QueryResult<()> {
        let mut estimator = self
            .cardinality_estimator
            .write()
            .map_err(|_| QueryError::ExecutionError("Estimator lock poisoned".to_string()))?;
        estimator.update_table_stats(table, row_count);
        Ok(())
    }

    /// Register an existing index
    pub fn register_index(&self, table: &str, columns: Vec<String>) -> QueryResult<()> {
        let mut advisor = self
            .index_advisor
            .write()
            .map_err(|_| QueryError::ExecutionError("Advisor lock poisoned".to_string()))?;
        advisor.register_index(table, columns);
        Ok(())
    }

    /// Clear the plan cache
    pub fn clear_cache(&self) -> QueryResult<()> {
        let mut cache = self
            .cache
            .write()
            .map_err(|_| QueryError::ExecutionError("Cache lock poisoned".to_string()))?;
        cache.clear();
        Ok(())
    }

    /// Cleanup stale cache entries
    pub fn cleanup(&self) -> QueryResult<()> {
        let mut cache = self
            .cache
            .write()
            .map_err(|_| QueryError::ExecutionError("Cache lock poisoned".to_string()))?;
        cache.cleanup_stale();
        Ok(())
    }
}

impl Default for AdaptiveOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Value;

    #[test]
    fn test_query_fingerprint() {
        let query1 = Query::select("users")
            .columns(vec!["id", "name"])
            .filter(Expression::eq(
                Expression::column("id"),
                Expression::literal(Value::Int(1)),
            ));

        let query2 = Query::select("users")
            .columns(vec!["id", "name"])
            .filter(Expression::eq(
                Expression::column("id"),
                Expression::literal(Value::Int(2)),
            ));

        let fp1 = QueryFingerprint::from_query(&query1);
        let fp2 = QueryFingerprint::from_query(&query2);

        // Same structure, different values should have same fingerprint
        assert_eq!(fp1.hash, fp2.hash);
    }

    #[test]
    fn test_plan_statistics() {
        let mut stats = PlanStatistics::new();

        stats.record_execution(Duration::from_millis(100), 50, 100.0, 100);
        stats.record_execution(Duration::from_millis(120), 55, 100.0, 100);
        stats.record_execution(Duration::from_millis(90), 45, 100.0, 100);

        assert_eq!(stats.execution_count, 3);
        assert!(stats.min_execution_time <= stats.avg_execution_time);
        assert!(stats.avg_execution_time <= stats.max_execution_time);
    }

    #[test]
    fn test_plan_cache_lru() {
        let config = PlanCacheConfig {
            max_entries: 3,
            ..Default::default()
        };
        let mut cache = PlanCache::with_config(config);

        // Insert 3 entries
        for i in 0..3 {
            let query = Query::select(&format!("table{}", i));
            let fp = QueryFingerprint::from_query(&query);
            let plan = PlanNode::scan(&format!("table{}", i));
            cache.insert(CachedPlanEntry::new(fp, plan));
        }

        assert_eq!(cache.len(), 3);

        // Insert 4th entry, should evict one
        let query = Query::select("table3");
        let fp = QueryFingerprint::from_query(&query);
        let plan = PlanNode::scan("table3");
        cache.insert(CachedPlanEntry::new(fp, plan));

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.statistics().evictions, 1);
    }

    #[test]
    fn test_cardinality_estimator() {
        let mut estimator = CardinalityEstimator::new();

        estimator.update_table_stats("users", 10000);
        estimator.update_column_stats("users", "status", 5);

        let scan = PlanNode::Scan {
            table: "users".to_string(),
            columns: vec!["id".to_string()],
            filter: Some(Expression::eq(
                Expression::column("status"),
                Expression::literal(Value::String("active".to_string())),
            )),
        };

        let estimate = estimator.estimate(&scan);
        // With 5 distinct values, selectivity should be ~0.2, so ~2000 rows
        assert!(estimate > 0 && estimate < 10000);
    }

    #[test]
    fn test_cardinality_feedback() {
        let mut estimator = CardinalityEstimator::new();

        // Initial estimate
        estimator.update_table_stats("orders", 1000);

        // Record feedback - actual was higher than estimated
        for _ in 0..10 {
            estimator.feedback("orders", 100, 200);
        }

        // Correction factor should have increased
        let factor = estimator
            .correction_factors
            .get("orders")
            .copied()
            .unwrap_or(1.0);
        assert!(factor > 1.0);
    }

    #[test]
    fn test_index_advisor() {
        let mut advisor = IndexAdvisor::new();

        // Record queries with slow execution times
        for _ in 0..20 {
            let query = Query::select("orders").filter(Expression::eq(
                Expression::column("customer_id"),
                Expression::literal(Value::Int(1)),
            ));
            advisor.record_query(&query, Duration::from_millis(200));
        }

        let recommendations = advisor.recommend();
        assert!(!recommendations.is_empty());
        assert!(
            recommendations
                .iter()
                .any(|r| r.columns.contains(&"customer_id".to_string()))
        );
    }

    #[test]
    fn test_cost_model() {
        let mut cost_model = CostModel::new();

        let plan = PlanNode::Scan {
            table: "users".to_string(),
            columns: vec![],
            filter: None,
        };

        let initial_cost = cost_model.estimate(&plan, 1000);
        assert!(initial_cost > 0.0);

        // Provide feedback that actual was slower
        cost_model.feedback(
            initial_cost,
            Duration::from_millis((initial_cost * 2.0) as u64),
        );

        // Coefficients should have increased
        let new_cost = cost_model.estimate(&plan, 1000);
        assert!(new_cost > initial_cost);
    }

    #[test]
    fn test_plan_selector_ucb() {
        let mut selector = PlanSelector::new();

        let mut arms = vec![
            PlanSelector::create_arm(0),
            PlanSelector::create_arm(1),
            PlanSelector::create_arm(2),
        ];

        // Initially should explore all arms
        for _ in 0..15 {
            let selected = selector.select(&mut arms);
            PlanSelector::update_arm(&mut arms[selected], Duration::from_millis(100));
        }

        // All arms should have been selected at least once
        assert!(arms.iter().all(|a| a.selections > 0));
    }

    #[test]
    fn test_plan_selector_exploitation() {
        let mut selector = PlanSelector::new();

        let mut arms = vec![PlanSelector::create_arm(0), PlanSelector::create_arm(1)];

        // Simulate: arm 0 is faster
        for _ in 0..5 {
            PlanSelector::update_arm(&mut arms[0], Duration::from_millis(50));
            PlanSelector::update_arm(&mut arms[1], Duration::from_millis(200));
        }

        // After initial exploration, should favor arm 0
        let mut arm0_selections = 0;
        for _ in 0..20 {
            let selected = selector.select(&mut arms);
            if selected == 0 {
                arm0_selections += 1;
            }
            PlanSelector::update_arm(
                &mut arms[selected],
                if selected == 0 {
                    Duration::from_millis(50)
                } else {
                    Duration::from_millis(200)
                },
            );
        }

        // Arm 0 should be selected more often
        assert!(arm0_selections > 10);
    }

    #[test]
    fn test_adaptive_optimizer_basic() {
        let optimizer = AdaptiveOptimizer::new();

        let query = Query::select("users")
            .columns(vec!["id", "name"])
            .filter(Expression::eq(
                Expression::column("id"),
                Expression::literal(Value::Int(1)),
            ));

        let plan = PlanNode::Scan {
            table: "users".to_string(),
            columns: vec!["id".to_string(), "name".to_string()],
            filter: query.filter.clone(),
        };

        // First optimization should cache the plan
        let result = optimizer.optimize(&query, plan.clone()).unwrap();

        // Second call should return cached plan
        let cached = optimizer.optimize(&query, plan).unwrap();

        let stats = optimizer.cache_statistics().unwrap();
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn test_adaptive_optimizer_feedback() {
        let optimizer = AdaptiveOptimizer::new();

        let query = Query::select("users");
        let plan = PlanNode::scan("users");

        optimizer.optimize(&query, plan).unwrap();

        // Record feedback
        optimizer
            .feedback(&query, Duration::from_millis(100), 500)
            .unwrap();
        optimizer
            .feedback(&query, Duration::from_millis(120), 510)
            .unwrap();

        let stats = optimizer.cache_statistics().unwrap();
        assert!(stats.hits > 0);
    }

    #[test]
    fn test_regression_detection() {
        let mut stats = PlanStatistics::new();

        // Record consistent execution times
        for _ in 0..10 {
            stats.record_execution(Duration::from_millis(100), 100, 100.0, 100);
        }

        assert!(!stats.is_degrading());

        // Record degraded execution times
        for _ in 0..5 {
            stats.record_execution(Duration::from_millis(300), 100, 100.0, 100);
        }

        assert!(stats.is_degrading());
    }

    #[test]
    fn test_cache_hit_ratio() {
        let mut cache = PlanCache::new();

        let query = Query::select("users");
        let fp = QueryFingerprint::from_query(&query);
        let plan = PlanNode::scan("users");

        cache.insert(CachedPlanEntry::new(fp.clone(), plan));

        // Miss
        let other_query = Query::select("orders");
        let other_fp = QueryFingerprint::from_query(&other_query);
        let _ = cache.get(&other_fp);

        // Hits
        let _ = cache.get(&fp);
        let _ = cache.get(&fp);

        let stats = cache.statistics();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_ratio() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_runtime_feedback_loop() {
        let mut feedback = RuntimeFeedbackLoop::new(RuntimeFeedbackConfig::default());

        let fp = QueryFingerprint {
            hash: 12345,
            description: "SELECT * FROM users WHERE id = ?".to_string(),
            tables: vec!["users".to_string()],
            columns: vec!["id".to_string()],
        };

        // Record enough executions to exceed min_samples (default: 5)
        feedback.record_execution(&fp, 100, 50);
        feedback.record_execution(&fp, 80, 52);
        feedback.record_execution(&fp, 120, 48);
        feedback.record_execution(&fp, 90, 55);
        feedback.record_execution(&fp, 110, 45);

        let correction = feedback.get_cardinality_correction(&fp);
        assert!(correction.is_some());
        assert!(correction.unwrap() > 0.0);
    }
}

// ============================================================================
// AI-Driven Runtime Feedback Loop
// ============================================================================

/// Runtime feedback loop configuration.
///
/// Based on "AI-Driven Autonomous Database Management" (2025) and
/// "How AI is Transforming SQL Query Optimization" (2025):
/// - Tracks actual vs. estimated row counts per query fingerprint
/// - Uses exponential moving average to learn correction factors
/// - Automatically corrects cardinality estimators based on runtime data
/// - Learns optimal execution path preferences (HDC/columnar/B-tree)
#[derive(Debug, Clone)]
pub struct RuntimeFeedbackConfig {
    /// Smoothing factor for exponential moving average (0-1, default: 0.3)
    pub ema_alpha: f64,
    /// Minimum executions before applying corrections (default: 5)
    pub min_samples: usize,
    /// Maximum correction factor (to prevent runaway corrections, default: 100.0)
    pub max_correction_factor: f64,
    /// History window size per fingerprint (default: 100)
    pub history_window: usize,
}

impl Default for RuntimeFeedbackConfig {
    fn default() -> Self {
        Self {
            ema_alpha: 0.3,
            min_samples: 5,
            max_correction_factor: 100.0,
            history_window: 100,
        }
    }
}

/// Per-fingerprint execution history for the feedback loop
#[derive(Debug, Clone)]
struct FeedbackEntry {
    /// Estimated row count from the planner
    estimated_rows: VecDeque<usize>,
    /// Actual row count from execution
    actual_rows: VecDeque<usize>,
    /// EMA of correction ratio (actual / estimated)
    correction_ema: f64,
    /// Total executions recorded
    total_executions: u64,
    /// Sum of execution times in microseconds
    total_execution_us: u64,
}

impl FeedbackEntry {
    fn new() -> Self {
        Self {
            estimated_rows: VecDeque::new(),
            actual_rows: VecDeque::new(),
            correction_ema: 1.0,
            total_executions: 0,
            total_execution_us: 0,
        }
    }
}

/// AI-driven runtime feedback loop for query optimization.
///
/// Implements a lightweight feedback mechanism that learns from query execution
/// to improve future query plans. This is inspired by SQL Server 2025's
/// "Intelligent Query Processing" and Azure SQL's ML-based query optimizer.
///
/// Key features:
/// - **Cardinality correction**: Learns correction factors for estimated row counts
/// - **Execution path learning**: Tracks which execution path performs best per query
/// - **Regression detection**: Identifies when plan quality degrades over time
/// - **Auto-tuning**: Adjusts optimizer parameters based on workload patterns
pub struct RuntimeFeedbackLoop {
    /// Configuration
    config: RuntimeFeedbackConfig,
    /// Per-fingerprint feedback entries
    entries: HashMap<u64, FeedbackEntry>,
    /// Execution path preferences learned from runtime: fingerprint -> preferred path
    path_preferences: HashMap<u64, ExecutionPathPreference>,
}

/// Learned execution path preference
#[derive(Debug, Clone)]
pub struct ExecutionPathPreference {
    /// Path name (e.g., "btree", "columnar", "holographic")
    pub preferred_path: String,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Average execution time for preferred path (microseconds)
    pub avg_preferred_us: f64,
    /// Average execution time for other paths (microseconds)
    pub avg_other_us: f64,
}

impl RuntimeFeedbackLoop {
    /// Create a new feedback loop with default configuration
    pub fn new(config: RuntimeFeedbackConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            path_preferences: HashMap::new(),
        }
    }

    /// Record an execution result for a query fingerprint
    pub fn record_execution(
        &mut self,
        fingerprint: &QueryFingerprint,
        estimated_rows: usize,
        actual_rows: usize,
    ) {
        let entry = self
            .entries
            .entry(fingerprint.hash)
            .or_insert_with(FeedbackEntry::new);

        // Add to history windows
        entry.estimated_rows.push_back(estimated_rows);
        entry.actual_rows.push_back(actual_rows);
        entry.total_executions += 1;

        // Trim history
        while entry.estimated_rows.len() > self.config.history_window {
            entry.estimated_rows.pop_front();
            entry.actual_rows.pop_front();
        }

        // Update EMA of correction ratio
        let ratio = if estimated_rows > 0 {
            (actual_rows as f64 / estimated_rows as f64)
                .min(self.config.max_correction_factor)
                .max(1.0 / self.config.max_correction_factor)
        } else {
            1.0
        };

        entry.correction_ema =
            self.config.ema_alpha * ratio + (1.0 - self.config.ema_alpha) * entry.correction_ema;
    }

    /// Record execution time for a specific path
    pub fn record_path_execution(
        &mut self,
        fingerprint: &QueryFingerprint,
        path_name: &str,
        execution_us: u64,
    ) {
        let entry = self
            .entries
            .entry(fingerprint.hash)
            .or_insert_with(FeedbackEntry::new);
        entry.total_execution_us += execution_us;

        // Update path preference
        let pref = self
            .path_preferences
            .entry(fingerprint.hash)
            .or_insert_with(|| ExecutionPathPreference {
                preferred_path: path_name.to_string(),
                confidence: 0.5,
                avg_preferred_us: execution_us as f64,
                avg_other_us: execution_us as f64,
            });

        if pref.preferred_path == path_name {
            // Update average for preferred path
            let n = entry.total_executions as f64;
            pref.avg_preferred_us = pref.avg_preferred_us * (n - 1.0) / n + execution_us as f64 / n;
        } else {
            // Update average for other path
            let n = entry.total_executions as f64;
            pref.avg_other_us = pref.avg_other_us * (n - 1.0) / n + execution_us as f64 / n;

            // If this path is consistently better, switch preference
            if pref.avg_other_us < pref.avg_preferred_us * 0.8 {
                pref.preferred_path = path_name.to_string();
                std::mem::swap(&mut pref.avg_preferred_us, &mut pref.avg_other_us);
                pref.confidence = 0.6;
            }
        }

        // Update confidence based on sample size
        let samples = entry.total_executions.min(100) as f64;
        pref.confidence = (samples / 20.0).min(1.0);
    }

    /// Get the cardinality correction factor for a query fingerprint.
    ///
    /// Returns None if insufficient data, otherwise returns a multiplier
    /// to apply to the estimated row count.
    pub fn get_cardinality_correction(&self, fingerprint: &QueryFingerprint) -> Option<f64> {
        let entry = self.entries.get(&fingerprint.hash)?;

        if entry.total_executions < self.config.min_samples as u64 {
            return None;
        }

        Some(entry.correction_ema)
    }

    /// Get the preferred execution path for a query fingerprint
    pub fn get_path_preference(
        &self,
        fingerprint: &QueryFingerprint,
    ) -> Option<&ExecutionPathPreference> {
        self.path_preferences.get(&fingerprint.hash)
    }

    /// Get the number of tracked fingerprints
    pub fn tracked_fingerprints(&self) -> usize {
        self.entries.len()
    }

    /// Get summary statistics for the feedback loop
    pub fn summary(&self) -> FeedbackSummary {
        let total_entries = self.entries.len();
        let total_executions: u64 = self.entries.values().map(|e| e.total_executions).sum();
        let entries_with_correction = self
            .entries
            .values()
            .filter(|e| e.total_executions >= self.config.min_samples as u64)
            .count();

        let avg_correction = if entries_with_correction > 0 {
            self.entries
                .values()
                .filter(|e| e.total_executions >= self.config.min_samples as u64)
                .map(|e| e.correction_ema)
                .sum::<f64>()
                / entries_with_correction as f64
        } else {
            1.0
        };

        FeedbackSummary {
            total_fingerprints: total_entries,
            total_executions,
            fingerprints_with_corrections: entries_with_correction,
            avg_correction_factor: avg_correction,
            path_preferences_learned: self.path_preferences.len(),
        }
    }
}

/// Summary of feedback loop state
#[derive(Debug, Clone)]
pub struct FeedbackSummary {
    /// Total unique query fingerprints tracked
    pub total_fingerprints: usize,
    /// Total executions recorded
    pub total_executions: u64,
    /// Number of fingerprints with enough data for corrections
    pub fingerprints_with_corrections: usize,
    /// Average correction factor across all fingerprints
    pub avg_correction_factor: f64,
    /// Number of execution path preferences learned
    pub path_preferences_learned: usize,
}
