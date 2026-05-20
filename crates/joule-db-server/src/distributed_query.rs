//! Distributed Query Executor for JouleDB
//!
//! This module implements distributed SQL query execution across multiple shards.
//!
//! ## Features
//!
//! - Automatic shard key extraction from WHERE clauses
//! - Query routing to appropriate shards based on sharding keys
//! - Scatter-gather execution for queries that span multiple shards
//! - Two-phase distributed aggregation (map on shards, reduce at coordinator)
//! - Cross-shard JOIN coordination
//! - Consistent read support via quorum reads
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │                          DistributedQueryExecutor                          │
//! │                                                                            │
//! │  1. Parse SQL → 2. Analyze Query → 3. Route to Shards → 4. Merge Results  │
//! └──────────────────────────────┬─────────────────────────────────────────────┘
//!                                │
//!       ┌────────────────────────┼────────────────────────────────┐
//!       ▼                        ▼                                ▼
//! ┌──────────────┐        ┌──────────────┐                ┌──────────────┐
//! │   Shard 0    │        │   Shard 1    │                │   Shard N    │
//! │  Execute +   │        │  Execute +   │      ...       │  Execute +   │
//! │  Partial Agg │        │  Partial Agg │                │  Partial Agg │
//! └──────────────┘        └──────────────┘                └──────────────┘
//! ```

use crate::query::{QueryErrorResponse, QueryExecutor, QueryRequest, QueryResponse};
use crate::sharding::{
    ConsistentHashRing, CrossShardCoordinator, CrossShardResult, KeyRange, Shard, ShardRouter,
    ShardingConfig, ShardingError, ShardingResult,
};
use joule_db_query::sql::{FromSource, SqlParser, SqlQuery, SqlStatement};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for distributed query execution
#[derive(Debug, Clone)]
pub struct DistributedQueryConfig {
    /// Maximum time to wait for shard responses
    pub query_timeout: Duration,
    /// Maximum parallel shard queries
    pub max_parallel_shards: usize,
    /// Whether to enable consistent reads (quorum)
    pub consistent_reads: bool,
    /// Read quorum size (for consistent reads)
    pub read_quorum: usize,
    /// Column name used as the default sharding key
    pub default_shard_key: String,
    /// Enable query result caching
    pub enable_result_cache: bool,
    /// Maximum cache size in entries
    pub max_cache_entries: usize,
    /// Cache TTL
    pub cache_ttl: Duration,
}

impl Default for DistributedQueryConfig {
    fn default() -> Self {
        Self {
            query_timeout: Duration::from_secs(30),
            max_parallel_shards: 16,
            consistent_reads: false,
            read_quorum: 2,
            default_shard_key: "id".to_string(),
            enable_result_cache: true,
            max_cache_entries: 10000,
            cache_ttl: Duration::from_secs(60),
        }
    }
}

// ============================================================================
// Query Analysis
// ============================================================================

/// Result of analyzing a query for sharding
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    /// Type of query operation
    pub query_type: QueryType,
    /// Tables involved in the query
    pub tables: Vec<String>,
    /// Shard keys extracted from WHERE clause (if any)
    pub shard_keys: Vec<ShardKey>,
    /// Whether this query can be targeted to specific shards
    pub is_targeted: bool,
    /// Whether this query requires aggregation across shards
    pub requires_aggregation: bool,
    /// Aggregation functions to merge
    pub aggregations: Vec<AggregateSpec>,
    /// Whether the query has ORDER BY (requires merge-sort)
    pub has_ordering: bool,
    /// ORDER BY columns and directions
    pub order_by: Vec<(String, bool)>, // (column, is_ascending)
    /// LIMIT value (if any)
    pub limit: Option<usize>,
    /// OFFSET value (if any)
    pub offset: Option<usize>,
}

/// Type of query for execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Point query with exact shard key
    PointRead,
    /// Range scan that may span shards
    RangeScan,
    /// Full table scan (all shards)
    FullScan,
    /// Insert operation
    Insert,
    /// Update operation
    Update,
    /// Delete operation
    Delete,
    /// DDL operation (CREATE, DROP, ALTER)
    Ddl,
}

/// A shard key extracted from the query
#[derive(Debug, Clone)]
pub struct ShardKey {
    /// Column name
    pub column: String,
    /// Key value (if exact match)
    pub value: Option<ShardKeyValue>,
    /// Key range (if range query)
    pub range: Option<(Option<ShardKeyValue>, Option<ShardKeyValue>)>,
}

/// Value types for shard keys
#[derive(Debug, Clone)]
pub enum ShardKeyValue {
    Int(i64),
    String(String),
    Bytes(Vec<u8>),
}

impl ShardKeyValue {
    /// Convert to bytes for consistent hashing
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            ShardKeyValue::Int(i) => i.to_be_bytes().to_vec(),
            ShardKeyValue::String(s) => s.as_bytes().to_vec(),
            ShardKeyValue::Bytes(b) => b.clone(),
        }
    }
}

/// Specification for an aggregate function that needs distributed computation
#[derive(Debug, Clone)]
pub struct AggregateSpec {
    /// Function name (SUM, COUNT, AVG, MIN, MAX)
    pub function: String,
    /// Column being aggregated
    pub column: Option<String>,
    /// Alias for the result
    pub alias: String,
}

// ============================================================================
// Shard Query Result
// ============================================================================

/// Result from a single shard query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardQueryResult {
    /// Shard ID that produced this result
    pub shard_id: String,
    /// Column names
    pub columns: Vec<String>,
    /// Rows of data
    pub rows: Vec<Vec<serde_json::Value>>,
    /// Affected rows (for writes)
    pub affected_rows: Option<usize>,
    /// Execution time on this shard
    pub execution_time_ms: u64,
    /// Partial aggregates (for distributed aggregation)
    pub partial_aggregates: HashMap<String, PartialAggregate>,
}

/// Partial aggregate state from a shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialAggregate {
    /// Function type
    pub function: String,
    /// Running sum (for SUM, AVG)
    pub sum: Option<f64>,
    /// Count (for COUNT, AVG)
    pub count: Option<u64>,
    /// Min value
    pub min: Option<serde_json::Value>,
    /// Max value
    pub max: Option<serde_json::Value>,
}

// ============================================================================
// Distributed Query Executor
// ============================================================================

/// Shard-local executor trait for executing queries on individual shards
pub trait ShardExecutor: Send + Sync {
    /// Execute a query on this shard
    fn execute_on_shard(
        &self,
        shard_id: &str,
        request: &QueryRequest,
    ) -> Result<ShardQueryResult, QueryErrorResponse>;
}

/// Distributed query executor that routes queries across shards
pub struct DistributedQueryExecutor<E: ShardExecutor> {
    /// Configuration
    config: DistributedQueryConfig,
    /// Shard router for key-to-shard mapping
    router: Arc<ShardRouter>,
    /// Cross-shard coordinator for scatter-gather
    coordinator: Arc<CrossShardCoordinator>,
    /// Shard-local executor
    shard_executor: Arc<E>,
    /// Query cache
    cache: RwLock<HashMap<String, CacheEntry>>,
    /// Statistics
    stats: RwLock<DistributedQueryStats>,
}

/// Cache entry for query results
#[derive(Clone)]
struct CacheEntry {
    response: QueryResponse,
    inserted_at: Instant,
}

/// Statistics for distributed query execution
#[derive(Debug, Clone, Default)]
pub struct DistributedQueryStats {
    /// Total queries executed
    pub total_queries: u64,
    /// Targeted queries (single shard)
    pub targeted_queries: u64,
    /// Scatter-gather queries (multiple shards)
    pub scatter_gather_queries: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Average query latency in ms
    pub avg_latency_ms: f64,
    /// Total rows processed
    pub total_rows_processed: u64,
}

impl<E: ShardExecutor> DistributedQueryExecutor<E> {
    /// Create a new distributed query executor
    pub fn new(
        config: DistributedQueryConfig,
        router: Arc<ShardRouter>,
        shard_executor: Arc<E>,
    ) -> Self {
        let coordinator = Arc::new(CrossShardCoordinator::new(
            router.clone(),
            config.query_timeout,
            config.max_parallel_shards,
        ));

        Self {
            config,
            router,
            coordinator,
            shard_executor,
            cache: RwLock::new(HashMap::new()),
            stats: RwLock::new(DistributedQueryStats::default()),
        }
    }

    /// Get current statistics
    pub async fn stats(&self) -> DistributedQueryStats {
        self.stats.read().await.clone()
    }

    /// Clear the query cache
    pub async fn clear_cache(&self) {
        self.cache.write().await.clear();
    }

    /// Analyze a SQL statement to determine execution strategy
    fn analyze_query(&self, stmt: &SqlStatement) -> QueryAnalysis {
        match stmt {
            SqlStatement::Select(query) => self.analyze_select(query),
            SqlStatement::Insert(insert) => QueryAnalysis {
                query_type: QueryType::Insert,
                tables: vec![insert.table.clone()],
                shard_keys: Vec::new(),
                is_targeted: false,
                requires_aggregation: false,
                aggregations: Vec::new(),
                has_ordering: false,
                order_by: Vec::new(),
                limit: None,
                offset: None,
            },
            SqlStatement::Update(update) => {
                let shard_keys = self.extract_shard_keys(&update.where_clause);
                QueryAnalysis {
                    query_type: QueryType::Update,
                    tables: vec![update.table.clone()],
                    shard_keys: shard_keys.clone(),
                    is_targeted: !shard_keys.is_empty()
                        && shard_keys.iter().any(|k| k.value.is_some()),
                    requires_aggregation: false,
                    aggregations: Vec::new(),
                    has_ordering: false,
                    order_by: Vec::new(),
                    limit: None,
                    offset: None,
                }
            }
            SqlStatement::Delete(delete) => {
                let shard_keys = self.extract_shard_keys(&delete.where_clause);
                QueryAnalysis {
                    query_type: QueryType::Delete,
                    tables: vec![delete.table.clone()],
                    shard_keys: shard_keys.clone(),
                    is_targeted: !shard_keys.is_empty()
                        && shard_keys.iter().any(|k| k.value.is_some()),
                    requires_aggregation: false,
                    aggregations: Vec::new(),
                    has_ordering: false,
                    order_by: Vec::new(),
                    limit: None,
                    offset: None,
                }
            }
            SqlStatement::CreateTable(_) | SqlStatement::DropTable { .. } => QueryAnalysis {
                query_type: QueryType::Ddl,
                tables: Vec::new(),
                shard_keys: Vec::new(),
                is_targeted: false,
                requires_aggregation: false,
                aggregations: Vec::new(),
                has_ordering: false,
                order_by: Vec::new(),
                limit: None,
                offset: None,
            },
            _ => QueryAnalysis {
                query_type: QueryType::FullScan,
                tables: Vec::new(),
                shard_keys: Vec::new(),
                is_targeted: false,
                requires_aggregation: false,
                aggregations: Vec::new(),
                has_ordering: false,
                order_by: Vec::new(),
                limit: None,
                offset: None,
            },
        }
    }

    /// Analyze a SELECT query
    fn analyze_select(&self, query: &SqlQuery) -> QueryAnalysis {
        // Extract tables
        let tables = self.extract_tables(query);

        // Extract shard keys from WHERE clause
        let shard_keys = self.extract_shard_keys(&query.where_clause);

        // Check for aggregations
        let (requires_aggregation, aggregations) = self.analyze_aggregations(query);

        // Check for ordering
        let has_ordering = !query.order_by.is_empty();
        let order_by = query
            .order_by
            .iter()
            .map(|ob| {
                let col = match &ob.expr {
                    joule_db_query::ast::Expression::Column(name) => name.clone(),
                    _ => format!("{:?}", ob.expr),
                };
                (col, !ob.descending)
            })
            .collect();

        // Determine query type
        let query_type = if shard_keys
            .iter()
            .any(|k| k.value.is_some() && k.column == self.config.default_shard_key)
        {
            QueryType::PointRead
        } else if shard_keys.iter().any(|k| k.range.is_some()) {
            QueryType::RangeScan
        } else {
            QueryType::FullScan
        };

        let is_targeted = matches!(query_type, QueryType::PointRead);

        QueryAnalysis {
            query_type,
            tables,
            shard_keys,
            is_targeted,
            requires_aggregation,
            aggregations,
            has_ordering,
            order_by,
            limit: query.limit,
            offset: query.offset,
        }
    }

    /// Extract table names from a query
    fn extract_tables(&self, query: &SqlQuery) -> Vec<String> {
        let mut tables = Vec::new();
        if let Some(ref from) = query.from {
            match &from.source {
                FromSource::Table(name) => tables.push(name.clone()),
                FromSource::Subquery(_) => {} // Subqueries handled recursively
            }
        }
        // Joins are on SqlQuery directly, not inside SqlFrom
        for join in &query.joins {
            tables.push(join.table.clone());
        }
        tables
    }

    /// Extract shard keys from WHERE clause
    fn extract_shard_keys(
        &self,
        where_clause: &Option<joule_db_query::ast::Expression>,
    ) -> Vec<ShardKey> {
        use joule_db_query::ast::{Expression, Operator, Value};

        let mut shard_keys = Vec::new();

        if let Some(expr) = where_clause {
            self.extract_keys_from_expression(expr, &mut shard_keys);
        }

        shard_keys
    }

    /// Recursively extract shard keys from an expression
    fn extract_keys_from_expression(
        &self,
        expr: &joule_db_query::ast::Expression,
        shard_keys: &mut Vec<ShardKey>,
    ) {
        use joule_db_query::ast::{Expression, Operator, Value};

        match expr {
            Expression::Binary { left, op, right } => match op {
                Operator::Eq => {
                    // Check for column = value pattern
                    if let (Expression::Column(col), Expression::Literal(val)) =
                        (left.as_ref(), right.as_ref())
                    {
                        let value = match val {
                            Value::Int(i) => Some(ShardKeyValue::Int(*i)),
                            Value::String(s) => Some(ShardKeyValue::String(s.clone())),
                            Value::Bytes(b) => Some(ShardKeyValue::Bytes(b.clone())),
                            _ => None,
                        };
                        if let Some(v) = value {
                            shard_keys.push(ShardKey {
                                column: col.clone(),
                                value: Some(v),
                                range: None,
                            });
                        }
                    }
                    // Also check value = column pattern
                    if let (Expression::Literal(val), Expression::Column(col)) =
                        (left.as_ref(), right.as_ref())
                    {
                        let value = match val {
                            Value::Int(i) => Some(ShardKeyValue::Int(*i)),
                            Value::String(s) => Some(ShardKeyValue::String(s.clone())),
                            Value::Bytes(b) => Some(ShardKeyValue::Bytes(b.clone())),
                            _ => None,
                        };
                        if let Some(v) = value {
                            shard_keys.push(ShardKey {
                                column: col.clone(),
                                value: Some(v),
                                range: None,
                            });
                        }
                    }
                }
                Operator::And => {
                    // Recurse into both sides
                    self.extract_keys_from_expression(left, shard_keys);
                    self.extract_keys_from_expression(right, shard_keys);
                }
                Operator::Gt | Operator::Ge | Operator::Lt | Operator::Le => {
                    // Range condition
                    if let (Expression::Column(col), Expression::Literal(val)) =
                        (left.as_ref(), right.as_ref())
                    {
                        let value = match val {
                            Value::Int(i) => Some(ShardKeyValue::Int(*i)),
                            Value::String(s) => Some(ShardKeyValue::String(s.clone())),
                            _ => None,
                        };
                        if let Some(v) = value {
                            let range = match op {
                                Operator::Gt | Operator::Ge => (Some(v), None),
                                Operator::Lt | Operator::Le => (None, Some(v)),
                                _ => return,
                            };
                            shard_keys.push(ShardKey {
                                column: col.clone(),
                                value: None,
                                range: Some(range),
                            });
                        }
                    }
                }
                _ => {}
            },
            Expression::In { expr, list, .. } => {
                // IN clause with literal values
                if let Expression::Column(col) = expr.as_ref() {
                    for val in list {
                        if let Expression::Literal(v) = val {
                            let value = match v {
                                Value::Int(i) => Some(ShardKeyValue::Int(*i)),
                                Value::String(s) => Some(ShardKeyValue::String(s.clone())),
                                _ => None,
                            };
                            if let Some(v) = value {
                                shard_keys.push(ShardKey {
                                    column: col.clone(),
                                    value: Some(v),
                                    range: None,
                                });
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Analyze aggregation functions in a query
    fn analyze_aggregations(&self, query: &SqlQuery) -> (bool, Vec<AggregateSpec>) {
        use joule_db_query::ast::Expression;

        let mut aggregations = Vec::new();

        for (idx, select_item) in query.columns.iter().enumerate() {
            // SqlColumn has an expr field containing the Expression
            if let Expression::Function { name, args, .. } = &select_item.expr {
                let func_upper = name.to_uppercase();
                if matches!(func_upper.as_str(), "SUM" | "COUNT" | "AVG" | "MIN" | "MAX") {
                    let column = args.first().and_then(|e| {
                        if let Expression::Column(c) = e {
                            Some(c.clone())
                        } else {
                            None
                        }
                    });
                    aggregations.push(AggregateSpec {
                        function: func_upper,
                        column,
                        alias: select_item
                            .alias
                            .clone()
                            .unwrap_or_else(|| format!("agg_{}", idx)),
                    });
                }
            }
        }

        (!aggregations.is_empty(), aggregations)
    }

    /// Route a query to appropriate shards based on analysis
    fn route_query(&self, analysis: &QueryAnalysis) -> ShardingResult<Vec<String>> {
        match analysis.query_type {
            QueryType::PointRead => {
                // Find the specific shard for the key
                if let Some(key) = analysis
                    .shard_keys
                    .iter()
                    .find(|k| k.value.is_some() && k.column == self.config.default_shard_key)
                {
                    if let Some(ref value) = key.value {
                        let bytes = value.to_bytes();
                        let shard_id = self.router.route_key(&bytes)?.id.clone();
                        return Ok(vec![shard_id]);
                    }
                }
                // Fallback to all shards
                Ok(self.router.get_all_shard_ids())
            }
            QueryType::RangeScan => {
                // Extract range bounds from shard keys and route to overlapping shards
                if let Some(range_key) = analysis.shard_keys.iter().find(|k| k.range.is_some()) {
                    if let Some((ref start_opt, ref end_opt)) = range_key.range {
                        let start_bytes =
                            start_opt.as_ref().map(|v| v.to_bytes()).unwrap_or_default();
                        let end_bytes = end_opt
                            .as_ref()
                            .map(|v| v.to_bytes())
                            .unwrap_or_else(|| vec![0xFF; 8]);
                        match self.router.route_range(&start_bytes, &end_bytes) {
                            Ok(shards) => return Ok(shards.into_iter().map(|s| s.id).collect()),
                            Err(_) => return Ok(self.router.get_all_shard_ids()),
                        }
                    }
                }
                // Fallback: no extractable range bounds
                Ok(self.router.get_all_shard_ids())
            }
            QueryType::FullScan | QueryType::Ddl => {
                // DDL and full scans go to all shards
                Ok(self.router.get_all_shard_ids())
            }
            QueryType::Insert | QueryType::Update | QueryType::Delete => {
                // Writes need to go to the correct shard(s)
                if analysis.is_targeted {
                    if let Some(key) = analysis.shard_keys.iter().find(|k| k.value.is_some()) {
                        if let Some(ref value) = key.value {
                            let bytes = value.to_bytes();
                            let shard_id = self.router.route_key(&bytes)?.id.clone();
                            return Ok(vec![shard_id]);
                        }
                    }
                }
                // Non-targeted writes go to all shards
                Ok(self.router.get_all_shard_ids())
            }
        }
    }

    /// Execute a query across multiple shards and merge results
    fn execute_distributed(
        &self,
        request: &QueryRequest,
        shard_ids: &[String],
        analysis: &QueryAnalysis,
        start: Instant,
    ) -> Result<QueryResponse, QueryErrorResponse> {
        let mut all_results: Vec<ShardQueryResult> = Vec::new();
        let mut errors: Vec<(String, String)> = Vec::new();

        // Execute on each shard
        for shard_id in shard_ids {
            match self.shard_executor.execute_on_shard(shard_id, request) {
                Ok(result) => all_results.push(result),
                Err(e) => errors.push((shard_id.clone(), e.message.clone())),
            }
        }

        // If all shards failed, return error
        if all_results.is_empty() && !errors.is_empty() {
            return Err(QueryErrorResponse::execution_error(&format!(
                "All shards failed: {:?}",
                errors
            )));
        }

        // Merge results
        self.merge_results(all_results, analysis, start)
    }

    /// Merge results from multiple shards
    fn merge_results(
        &self,
        results: Vec<ShardQueryResult>,
        analysis: &QueryAnalysis,
        start: Instant,
    ) -> Result<QueryResponse, QueryErrorResponse> {
        if results.is_empty() {
            return Ok(QueryResponse {
                columns: Vec::new(),
                rows: Vec::new(),
                affected_rows: Some(0),
                execution_time_ms: start.elapsed().as_millis() as u64,
                truncated: false,
                warnings: Vec::new(),
                energy_joules: None,
                power_watts: None,
                device_target: None,
                algorithm_type: None,
                session_id: None,
                viz_hint: None,
            });
        }

        // Get columns from first result
        let columns = results[0].columns.clone();

        // Handle aggregation queries
        if analysis.requires_aggregation {
            return self.merge_aggregates(&results, &analysis.aggregations, &columns, start);
        }

        // Merge rows from all shards
        let mut merged_rows: Vec<Vec<serde_json::Value>> = Vec::new();
        let mut total_affected = 0usize;

        for result in &results {
            merged_rows.extend(result.rows.clone());
            if let Some(affected) = result.affected_rows {
                total_affected += affected;
            }
        }

        // Apply ordering if needed (merge-sort)
        if analysis.has_ordering && !analysis.order_by.is_empty() {
            self.sort_results(&mut merged_rows, &columns, &analysis.order_by);
        }

        // Apply OFFSET
        if let Some(offset) = analysis.offset {
            if offset < merged_rows.len() {
                merged_rows = merged_rows.into_iter().skip(offset).collect();
            } else {
                merged_rows.clear();
            }
        }

        // Apply LIMIT
        let truncated = if let Some(limit) = analysis.limit {
            if merged_rows.len() > limit {
                merged_rows.truncate(limit);
                true
            } else {
                false
            }
        } else {
            false
        };

        Ok(QueryResponse {
            columns,
            rows: merged_rows,
            affected_rows: if total_affected > 0 {
                Some(total_affected)
            } else {
                None
            },
            execution_time_ms: start.elapsed().as_millis() as u64,
            truncated,
            warnings: Vec::new(),
            energy_joules: None,
            power_watts: None,
            device_target: None,
            algorithm_type: None,
            session_id: None,
            viz_hint: None,
        })
    }

    /// Merge aggregate results using two-phase aggregation
    fn merge_aggregates(
        &self,
        results: &[ShardQueryResult],
        aggregations: &[AggregateSpec],
        columns: &[String],
        start: Instant,
    ) -> Result<QueryResponse, QueryErrorResponse> {
        let mut final_row: Vec<serde_json::Value> = Vec::new();

        for agg in aggregations {
            let merged_value = match agg.function.as_str() {
                "SUM" => {
                    let mut total: f64 = 0.0;
                    for result in results {
                        if let Some(partial) = result.partial_aggregates.get(&agg.alias) {
                            if let Some(sum) = partial.sum {
                                total += sum;
                            }
                        } else {
                            // Fallback: sum from rows
                            for row in &result.rows {
                                if let Some(idx) =
                                    result.columns.iter().position(|c| c == &agg.alias)
                                {
                                    if let Some(val) = row.get(idx) {
                                        if let Some(n) = val.as_f64() {
                                            total += n;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    serde_json::json!(total)
                }
                "COUNT" => {
                    let mut total: u64 = 0;
                    for result in results {
                        if let Some(partial) = result.partial_aggregates.get(&agg.alias) {
                            if let Some(count) = partial.count {
                                total += count;
                            }
                        } else {
                            // Fallback: count rows
                            total += result.rows.len() as u64;
                        }
                    }
                    serde_json::json!(total)
                }
                "AVG" => {
                    let mut total_sum: f64 = 0.0;
                    let mut total_count: u64 = 0;
                    for result in results {
                        if let Some(partial) = result.partial_aggregates.get(&agg.alias) {
                            if let (Some(sum), Some(count)) = (partial.sum, partial.count) {
                                total_sum += sum;
                                total_count += count;
                            }
                        }
                    }
                    if total_count > 0 {
                        serde_json::json!(total_sum / total_count as f64)
                    } else {
                        serde_json::Value::Null
                    }
                }
                "MIN" => {
                    let mut min_val: Option<serde_json::Value> = None;
                    for result in results {
                        if let Some(partial) = result.partial_aggregates.get(&agg.alias) {
                            if let Some(ref val) = partial.min {
                                match (&min_val, val) {
                                    (None, v) => min_val = Some(v.clone()),
                                    (Some(current), new) => {
                                        if self.json_compare(new, current)
                                            == std::cmp::Ordering::Less
                                        {
                                            min_val = Some(new.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    min_val.unwrap_or(serde_json::Value::Null)
                }
                "MAX" => {
                    let mut max_val: Option<serde_json::Value> = None;
                    for result in results {
                        if let Some(partial) = result.partial_aggregates.get(&agg.alias) {
                            if let Some(ref val) = partial.max {
                                match (&max_val, val) {
                                    (None, v) => max_val = Some(v.clone()),
                                    (Some(current), new) => {
                                        if self.json_compare(new, current)
                                            == std::cmp::Ordering::Greater
                                        {
                                            max_val = Some(new.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    max_val.unwrap_or(serde_json::Value::Null)
                }
                _ => serde_json::Value::Null,
            };
            final_row.push(merged_value);
        }

        let agg_columns: Vec<String> = aggregations.iter().map(|a| a.alias.clone()).collect();

        Ok(QueryResponse {
            columns: agg_columns,
            rows: vec![final_row],
            affected_rows: None,
            execution_time_ms: start.elapsed().as_millis() as u64,
            truncated: false,
            warnings: Vec::new(),
            energy_joules: None,
            power_watts: None,
            device_target: None,
            algorithm_type: None,
            session_id: None,
            viz_hint: None,
        })
    }

    /// Sort merged results
    fn sort_results(
        &self,
        rows: &mut Vec<Vec<serde_json::Value>>,
        columns: &[String],
        order_by: &[(String, bool)],
    ) {
        rows.sort_by(|a, b| {
            for (col, ascending) in order_by {
                if let Some(idx) = columns.iter().position(|c| c == col) {
                    let cmp = self.json_compare(
                        a.get(idx).unwrap_or(&serde_json::Value::Null),
                        b.get(idx).unwrap_or(&serde_json::Value::Null),
                    );
                    if cmp != std::cmp::Ordering::Equal {
                        return if *ascending { cmp } else { cmp.reverse() };
                    }
                }
            }
            std::cmp::Ordering::Equal
        });
    }

    /// Compare two JSON values
    fn json_compare(&self, a: &serde_json::Value, b: &serde_json::Value) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (a, b) {
            (serde_json::Value::Null, serde_json::Value::Null) => Ordering::Equal,
            (serde_json::Value::Null, _) => Ordering::Less,
            (_, serde_json::Value::Null) => Ordering::Greater,
            (serde_json::Value::Number(an), serde_json::Value::Number(bn)) => {
                let af = an.as_f64().unwrap_or(0.0);
                let bf = bn.as_f64().unwrap_or(0.0);
                af.partial_cmp(&bf).unwrap_or(Ordering::Equal)
            }
            (serde_json::Value::String(as_), serde_json::Value::String(bs)) => as_.cmp(bs),
            (serde_json::Value::Bool(ab), serde_json::Value::Bool(bb)) => ab.cmp(bb),
            _ => Ordering::Equal,
        }
    }

    /// Check cache for query result
    async fn check_cache(&self, cache_key: &str) -> Option<QueryResponse> {
        if !self.config.enable_result_cache {
            return None;
        }

        let cache = self.cache.read().await;
        if let Some(entry) = cache.get(cache_key) {
            if entry.inserted_at.elapsed() < self.config.cache_ttl {
                return Some(entry.response.clone());
            }
        }
        None
    }

    /// Store result in cache
    async fn store_cache(&self, cache_key: String, response: QueryResponse) {
        if !self.config.enable_result_cache {
            return;
        }

        let mut cache = self.cache.write().await;

        // Evict old entries if cache is full
        if cache.len() >= self.config.max_cache_entries {
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, v)| v.inserted_at)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }

        cache.insert(
            cache_key,
            CacheEntry {
                response,
                inserted_at: Instant::now(),
            },
        );
    }

    /// Generate cache key for a query
    fn cache_key(&self, request: &QueryRequest) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        request.sql.hash(&mut hasher);
        for arg in &request.args {
            arg.to_string().hash(&mut hasher);
        }
        format!("query_{:x}", hasher.finish())
    }

    /// Update statistics
    async fn update_stats(&self, is_targeted: bool, latency_ms: u64, rows: usize) {
        let mut stats = self.stats.write().await;
        stats.total_queries += 1;
        if is_targeted {
            stats.targeted_queries += 1;
        } else {
            stats.scatter_gather_queries += 1;
        }
        stats.total_rows_processed += rows as u64;

        // Update average latency (exponential moving average)
        let alpha = 0.1;
        stats.avg_latency_ms = (1.0 - alpha) * stats.avg_latency_ms + alpha * latency_ms as f64;
    }
}

impl<E: ShardExecutor + 'static> QueryExecutor for DistributedQueryExecutor<E> {
    fn execute(&self, request: &QueryRequest) -> Result<QueryResponse, QueryErrorResponse> {
        let start = Instant::now();

        // Check cache first
        let cache_key = self.cache_key(request);
        if let Some(cached) = futures::executor::block_on(self.check_cache(&cache_key)) {
            futures::executor::block_on(async {
                let mut stats = self.stats.write().await;
                stats.cache_hits += 1;
            });
            return Ok(cached);
        }
        futures::executor::block_on(async {
            let mut stats = self.stats.write().await;
            stats.cache_misses += 1;
        });

        // Parse the SQL
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse(&request.sql)
            .map_err(|e| QueryErrorResponse::syntax_error(&e.to_string(), 1, 1))?;

        // Analyze the query
        let analysis = self.analyze_query(&stmt);

        // Route to shards
        let shard_ids = self
            .route_query(&analysis)
            .map_err(|e| QueryErrorResponse::execution_error(&e.to_string()))?;

        // Execute
        let response = self.execute_distributed(request, &shard_ids, &analysis, start)?;

        // Update stats
        let latency = start.elapsed().as_millis() as u64;
        futures::executor::block_on(self.update_stats(
            analysis.is_targeted,
            latency,
            response.rows.len(),
        ));

        // Cache the result (only for reads)
        if matches!(
            analysis.query_type,
            QueryType::PointRead | QueryType::RangeScan | QueryType::FullScan
        ) {
            futures::executor::block_on(self.store_cache(cache_key, response.clone()));
        }

        Ok(response)
    }
}

// ============================================================================
// Simple Shard Executor Implementation
// ============================================================================

/// A simple shard executor that delegates to the local SimpleQueryExecutor
pub struct LocalShardExecutor {
    /// Mapping of shard_id to local executor
    executors: HashMap<String, Arc<crate::query::SimpleQueryExecutor>>,
}

impl LocalShardExecutor {
    /// Create a new local shard executor
    pub fn new() -> Self {
        Self {
            executors: HashMap::new(),
        }
    }

    /// Register an executor for a shard
    pub fn register_shard(
        &mut self,
        shard_id: String,
        executor: Arc<crate::query::SimpleQueryExecutor>,
    ) {
        self.executors.insert(shard_id, executor);
    }
}

impl Default for LocalShardExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ShardExecutor for LocalShardExecutor {
    fn execute_on_shard(
        &self,
        shard_id: &str,
        request: &QueryRequest,
    ) -> Result<ShardQueryResult, QueryErrorResponse> {
        let executor = self.executors.get(shard_id).ok_or_else(|| {
            QueryErrorResponse::execution_error(&format!("Shard {} not found", shard_id))
        })?;

        let start = Instant::now();
        let response = executor.execute(request)?;

        Ok(ShardQueryResult {
            shard_id: shard_id.to_string(),
            columns: response.columns,
            rows: response.rows,
            affected_rows: response.affected_rows,
            execution_time_ms: start.elapsed().as_millis() as u64,
            partial_aggregates: HashMap::new(),
        })
    }
}

// ============================================================================
// Remote Shard RPC Protocol
// ============================================================================

/// Shard RPC message types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardRpcType {
    /// Query request (JSON-encoded QueryRequest)
    Query = 0x00,
    /// Query result (JSON-encoded ShardQueryResult)
    Result = 0x01,
    /// Error response (UTF-8 error string)
    Error = 0x02,
}

impl ShardRpcType {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x00 => Some(Self::Query),
            0x01 => Some(Self::Result),
            0x02 => Some(Self::Error),
            _ => None,
        }
    }
}

/// Wire format for shard RPC messages:
/// - 4 bytes: payload length (little-endian u32)
/// - 1 byte: message type (ShardRpcType)
/// - N bytes: payload (JSON for Query/Result, UTF-8 for Error)
const SHARD_RPC_HEADER_SIZE: usize = 5;

/// Encode a shard RPC message
fn encode_shard_rpc(msg_type: ShardRpcType, payload: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(SHARD_RPC_HEADER_SIZE + payload.len());
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.push(msg_type as u8);
    buf.extend_from_slice(payload);
    buf
}

/// Decode a shard RPC message header (returns message type and payload length)
fn decode_shard_rpc_header(header: &[u8; SHARD_RPC_HEADER_SIZE]) -> Option<(ShardRpcType, usize)> {
    let len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let msg_type = ShardRpcType::from_u8(header[4])?;
    Some((msg_type, len))
}

// ============================================================================
// Remote Shard Executor
// ============================================================================

/// Executes queries on remote shard nodes via TCP.
///
/// Implements the `ShardExecutor` trait by serializing `QueryRequest` to JSON,
/// sending over TCP to the target shard's `ShardServer`, and deserializing
/// the response.
pub struct RemoteShardExecutor {
    /// Mapping of shard_id → remote address
    nodes: std::sync::RwLock<HashMap<String, std::net::SocketAddr>>,
    /// Connection timeout
    connect_timeout: Duration,
    /// Request timeout
    request_timeout: Duration,
}

impl RemoteShardExecutor {
    /// Create a new remote shard executor
    pub fn new(connect_timeout: Duration, request_timeout: Duration) -> Self {
        Self {
            nodes: std::sync::RwLock::new(HashMap::new()),
            connect_timeout,
            request_timeout,
        }
    }

    /// Register a remote shard node
    pub fn register_node(&self, shard_id: String, addr: std::net::SocketAddr) {
        crate::lock_util::write_lock(&self.nodes).insert(shard_id, addr);
    }

    /// Remove a remote shard node
    pub fn remove_node(&self, shard_id: &str) {
        crate::lock_util::write_lock(&self.nodes).remove(shard_id);
    }

    /// List registered nodes
    pub fn list_nodes(&self) -> HashMap<String, std::net::SocketAddr> {
        crate::lock_util::read_lock(&self.nodes).clone()
    }

    /// Send a query and receive the response over a TCP connection
    fn send_query(
        addr: std::net::SocketAddr,
        request: &QueryRequest,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Result<ShardQueryResult, QueryErrorResponse> {
        use std::io::{Read, Write};
        use std::net::TcpStream;

        // Connect with timeout
        let mut stream = TcpStream::connect_timeout(&addr, connect_timeout).map_err(|e| {
            QueryErrorResponse::execution_error(&format!(
                "Failed to connect to shard at {}: {}",
                addr, e
            ))
        })?;

        stream
            .set_read_timeout(Some(request_timeout))
            .map_err(|e| QueryErrorResponse::execution_error(&format!("Set timeout: {}", e)))?;
        stream
            .set_write_timeout(Some(request_timeout))
            .map_err(|e| QueryErrorResponse::execution_error(&format!("Set timeout: {}", e)))?;

        // Serialize and send query
        let payload = serde_json::to_vec(request)
            .map_err(|e| QueryErrorResponse::execution_error(&format!("Serialize: {}", e)))?;
        let msg = encode_shard_rpc(ShardRpcType::Query, &payload);
        stream.write_all(&msg).map_err(|e| {
            QueryErrorResponse::execution_error(&format!("Send to shard at {}: {}", addr, e))
        })?;

        // Read response header
        let mut header = [0u8; SHARD_RPC_HEADER_SIZE];
        stream.read_exact(&mut header).map_err(|e| {
            QueryErrorResponse::execution_error(&format!(
                "Read header from shard at {}: {}",
                addr, e
            ))
        })?;

        let (msg_type, payload_len) = decode_shard_rpc_header(&header)
            .ok_or_else(|| QueryErrorResponse::execution_error("Invalid shard RPC header"))?;

        // Read payload
        let mut payload_buf = vec![0u8; payload_len];
        stream.read_exact(&mut payload_buf).map_err(|e| {
            QueryErrorResponse::execution_error(&format!(
                "Read payload from shard at {}: {}",
                addr, e
            ))
        })?;

        match msg_type {
            ShardRpcType::Result => serde_json::from_slice(&payload_buf).map_err(|e| {
                QueryErrorResponse::execution_error(&format!("Deserialize shard result: {}", e))
            }),
            ShardRpcType::Error => {
                let error_msg = String::from_utf8_lossy(&payload_buf);
                Err(QueryErrorResponse::execution_error(&format!(
                    "Shard error: {}",
                    error_msg,
                )))
            }
            ShardRpcType::Query => Err(QueryErrorResponse::execution_error(
                "Unexpected Query message in response",
            )),
        }
    }
}

impl ShardExecutor for RemoteShardExecutor {
    fn execute_on_shard(
        &self,
        shard_id: &str,
        request: &QueryRequest,
    ) -> Result<ShardQueryResult, QueryErrorResponse> {
        let addr = {
            let nodes = crate::lock_util::read_lock(&self.nodes);
            *nodes.get(shard_id).ok_or_else(|| {
                QueryErrorResponse::execution_error(&format!("Unknown shard: {}", shard_id))
            })?
        };

        Self::send_query(addr, request, self.connect_timeout, self.request_timeout)
    }
}

// ============================================================================
// Shard Server (listens for and executes remote shard queries)
// ============================================================================

/// TCP server that accepts shard query requests from `RemoteShardExecutor`
/// instances and executes them locally.
pub struct ShardServer {
    /// Local query executor to handle incoming queries
    executor: Arc<dyn QueryExecutor>,
    /// Address to listen on
    listen_addr: std::net::SocketAddr,
}

impl ShardServer {
    /// Create a new shard server
    pub fn new(executor: Arc<dyn QueryExecutor>, listen_addr: std::net::SocketAddr) -> Self {
        Self {
            executor,
            listen_addr,
        }
    }

    /// Run the shard server (async, listens for TCP connections)
    pub async fn run(&self) -> Result<(), std::io::Error> {
        let listener = tokio::net::TcpListener::bind(self.listen_addr).await?;
        tracing::info!("ShardServer listening on {}", self.listen_addr);

        loop {
            let (stream, addr) = listener.accept().await?;
            let executor = self.executor.clone();
            tokio::spawn(async move {
                if let Err(e) = Self::handle_connection(stream, addr, executor).await {
                    tracing::warn!("ShardServer connection error from {}: {}", addr, e);
                }
            });
        }
    }

    /// Handle a single shard client connection
    async fn handle_connection(
        mut stream: tokio::net::TcpStream,
        addr: std::net::SocketAddr,
        executor: Arc<dyn QueryExecutor>,
    ) -> Result<(), std::io::Error> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        tracing::debug!("ShardServer accepted connection from {}", addr);

        loop {
            // Read header
            let mut header = [0u8; SHARD_RPC_HEADER_SIZE];
            match stream.read_exact(&mut header).await {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Client disconnected
                    return Ok(());
                }
                Err(e) => return Err(e),
            }

            let (msg_type, payload_len) = match decode_shard_rpc_header(&header) {
                Some(h) => h,
                None => {
                    let err_msg = b"Invalid RPC header";
                    let response = encode_shard_rpc(ShardRpcType::Error, err_msg);
                    stream.write_all(&response).await?;
                    continue;
                }
            };

            // Read payload
            let mut payload = vec![0u8; payload_len];
            stream.read_exact(&mut payload).await?;

            if msg_type != ShardRpcType::Query {
                let err_msg = b"Expected Query message";
                let response = encode_shard_rpc(ShardRpcType::Error, err_msg);
                stream.write_all(&response).await?;
                continue;
            }

            // Deserialize QueryRequest
            let request: QueryRequest = match serde_json::from_slice(&payload) {
                Ok(r) => r,
                Err(e) => {
                    let err_msg = format!("Invalid QueryRequest: {}", e);
                    let response = encode_shard_rpc(ShardRpcType::Error, err_msg.as_bytes());
                    stream.write_all(&response).await?;
                    continue;
                }
            };

            // Execute the query
            let start = Instant::now();
            match executor.execute(&request) {
                Ok(response) => {
                    let result = ShardQueryResult {
                        shard_id: "local".to_string(),
                        columns: response.columns,
                        rows: response.rows,
                        affected_rows: response.affected_rows,
                        execution_time_ms: start.elapsed().as_millis() as u64,
                        partial_aggregates: HashMap::new(),
                    };

                    let result_bytes = serde_json::to_vec(&result).unwrap_or_default();
                    let response_msg = encode_shard_rpc(ShardRpcType::Result, &result_bytes);
                    stream.write_all(&response_msg).await?;
                }
                Err(e) => {
                    let err_msg = format!("{}", e.message);
                    let response_msg = encode_shard_rpc(ShardRpcType::Error, err_msg.as_bytes());
                    stream.write_all(&response_msg).await?;
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_key_extraction() {
        use joule_db_query::ast::{Expression, Operator, Value};

        let config = DistributedQueryConfig::default();
        let router = Arc::new(ShardRouter::new(ShardingConfig::default()));

        // Create a mock executor
        struct MockExecutor;
        impl ShardExecutor for MockExecutor {
            fn execute_on_shard(
                &self,
                _shard_id: &str,
                _request: &QueryRequest,
            ) -> Result<ShardQueryResult, QueryErrorResponse> {
                Ok(ShardQueryResult {
                    shard_id: "test".to_string(),
                    columns: vec![],
                    rows: vec![],
                    affected_rows: None,
                    execution_time_ms: 0,
                    partial_aggregates: HashMap::new(),
                })
            }
        }

        let executor = DistributedQueryExecutor::new(config, router, Arc::new(MockExecutor));

        // Test simple equality
        let expr = Expression::Binary {
            left: Box::new(Expression::Column("id".to_string())),
            op: Operator::Eq,
            right: Box::new(Expression::Literal(Value::Int(42))),
        };

        let mut keys = Vec::new();
        executor.extract_keys_from_expression(&expr, &mut keys);

        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].column, "id");
        assert!(keys[0].value.is_some());
    }

    #[test]
    fn test_query_analysis() {
        let config = DistributedQueryConfig::default();
        let router = Arc::new(ShardRouter::new(ShardingConfig::default()));

        struct MockExecutor;
        impl ShardExecutor for MockExecutor {
            fn execute_on_shard(
                &self,
                _shard_id: &str,
                _request: &QueryRequest,
            ) -> Result<ShardQueryResult, QueryErrorResponse> {
                Ok(ShardQueryResult {
                    shard_id: "test".to_string(),
                    columns: vec![],
                    rows: vec![],
                    affected_rows: None,
                    execution_time_ms: 0,
                    partial_aggregates: HashMap::new(),
                })
            }
        }

        let executor = DistributedQueryExecutor::new(config, router, Arc::new(MockExecutor));

        // Parse a simple query
        let mut parser = SqlParser::new();
        let stmt = parser.parse("SELECT * FROM users WHERE id = 1").unwrap();

        let analysis = executor.analyze_query(&stmt);

        assert_eq!(analysis.query_type, QueryType::PointRead);
        assert!(analysis.is_targeted);
        assert!(!analysis.requires_aggregation);
    }

    #[test]
    fn test_aggregate_query_analysis() {
        let config = DistributedQueryConfig::default();
        let router = Arc::new(ShardRouter::new(ShardingConfig::default()));

        struct MockExecutor;
        impl ShardExecutor for MockExecutor {
            fn execute_on_shard(
                &self,
                _shard_id: &str,
                _request: &QueryRequest,
            ) -> Result<ShardQueryResult, QueryErrorResponse> {
                Ok(ShardQueryResult {
                    shard_id: "test".to_string(),
                    columns: vec![],
                    rows: vec![],
                    affected_rows: None,
                    execution_time_ms: 0,
                    partial_aggregates: HashMap::new(),
                })
            }
        }

        let executor = DistributedQueryExecutor::new(config, router, Arc::new(MockExecutor));

        // Parse an aggregate query
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT COUNT(*), SUM(amount) FROM orders")
            .unwrap();

        let analysis = executor.analyze_query(&stmt);

        assert_eq!(analysis.query_type, QueryType::FullScan);
        assert!(analysis.requires_aggregation);
        assert_eq!(analysis.aggregations.len(), 2);
    }

    #[test]
    fn test_json_compare() {
        let config = DistributedQueryConfig::default();
        let router = Arc::new(ShardRouter::new(ShardingConfig::default()));

        struct MockExecutor;
        impl ShardExecutor for MockExecutor {
            fn execute_on_shard(
                &self,
                _shard_id: &str,
                _request: &QueryRequest,
            ) -> Result<ShardQueryResult, QueryErrorResponse> {
                Ok(ShardQueryResult {
                    shard_id: "test".to_string(),
                    columns: vec![],
                    rows: vec![],
                    affected_rows: None,
                    execution_time_ms: 0,
                    partial_aggregates: HashMap::new(),
                })
            }
        }

        let executor = DistributedQueryExecutor::new(config, router, Arc::new(MockExecutor));

        assert_eq!(
            executor.json_compare(&serde_json::json!(1), &serde_json::json!(2)),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            executor.json_compare(&serde_json::json!("a"), &serde_json::json!("b")),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            executor.json_compare(&serde_json::Value::Null, &serde_json::json!(1)),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn test_merge_results() {
        let config = DistributedQueryConfig::default();
        let router = Arc::new(ShardRouter::new(ShardingConfig::default()));

        struct MockExecutor;
        impl ShardExecutor for MockExecutor {
            fn execute_on_shard(
                &self,
                _shard_id: &str,
                _request: &QueryRequest,
            ) -> Result<ShardQueryResult, QueryErrorResponse> {
                Ok(ShardQueryResult {
                    shard_id: "test".to_string(),
                    columns: vec![],
                    rows: vec![],
                    affected_rows: None,
                    execution_time_ms: 0,
                    partial_aggregates: HashMap::new(),
                })
            }
        }

        let executor = DistributedQueryExecutor::new(config, router, Arc::new(MockExecutor));

        let results = vec![
            ShardQueryResult {
                shard_id: "shard_0".to_string(),
                columns: vec!["id".to_string(), "name".to_string()],
                rows: vec![
                    vec![serde_json::json!(1), serde_json::json!("Alice")],
                    vec![serde_json::json!(2), serde_json::json!("Bob")],
                ],
                affected_rows: None,
                execution_time_ms: 10,
                partial_aggregates: HashMap::new(),
            },
            ShardQueryResult {
                shard_id: "shard_1".to_string(),
                columns: vec!["id".to_string(), "name".to_string()],
                rows: vec![vec![serde_json::json!(3), serde_json::json!("Charlie")]],
                affected_rows: None,
                execution_time_ms: 8,
                partial_aggregates: HashMap::new(),
            },
        ];

        let analysis = QueryAnalysis {
            query_type: QueryType::FullScan,
            tables: vec!["users".to_string()],
            shard_keys: vec![],
            is_targeted: false,
            requires_aggregation: false,
            aggregations: vec![],
            has_ordering: false,
            order_by: vec![],
            limit: None,
            offset: None,
        };

        let merged = executor
            .merge_results(results, &analysis, Instant::now())
            .unwrap();

        assert_eq!(merged.rows.len(), 3);
        assert_eq!(merged.columns, vec!["id", "name"]);
    }

    // ==================== Remote Shard Executor Tests ====================

    #[test]
    fn test_remote_shard_executor_register_node() {
        let executor = RemoteShardExecutor::new(Duration::from_secs(5), Duration::from_secs(30));

        let addr: std::net::SocketAddr = "127.0.0.1:9001".parse().unwrap();
        executor.register_node("shard-0".to_string(), addr);

        let nodes = executor.list_nodes();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes["shard-0"], addr);
    }

    #[test]
    fn test_remote_shard_executor_unknown_shard() {
        let executor = RemoteShardExecutor::new(Duration::from_secs(5), Duration::from_secs(30));

        let request = QueryRequest {
            sql: "SELECT 1".to_string(),
            params: std::collections::HashMap::new(),
            args: vec![],
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };

        let result = executor.execute_on_shard("nonexistent", &request);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.message.contains("Unknown shard"),
            "Got: {}",
            err.message
        );
    }

    #[test]
    fn test_shard_rpc_encode_decode() {
        let payload = b"hello world";
        let encoded = encode_shard_rpc(ShardRpcType::Query, payload);

        assert_eq!(encoded.len(), SHARD_RPC_HEADER_SIZE + payload.len());

        let mut header = [0u8; SHARD_RPC_HEADER_SIZE];
        header.copy_from_slice(&encoded[..SHARD_RPC_HEADER_SIZE]);

        let (msg_type, len) = decode_shard_rpc_header(&header).unwrap();
        assert_eq!(msg_type, ShardRpcType::Query);
        assert_eq!(len, payload.len());
        assert_eq!(&encoded[SHARD_RPC_HEADER_SIZE..], payload);
    }

    #[test]
    fn test_shard_rpc_types() {
        assert_eq!(ShardRpcType::from_u8(0x00), Some(ShardRpcType::Query));
        assert_eq!(ShardRpcType::from_u8(0x01), Some(ShardRpcType::Result));
        assert_eq!(ShardRpcType::from_u8(0x02), Some(ShardRpcType::Error));
        assert_eq!(ShardRpcType::from_u8(0xFF), None);
    }

    #[tokio::test]
    async fn test_shard_server_roundtrip() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // Create a simple mock executor
        struct MockQueryExecutor;
        impl QueryExecutor for MockQueryExecutor {
            fn execute(&self, request: &QueryRequest) -> Result<QueryResponse, QueryErrorResponse> {
                Ok(QueryResponse {
                    columns: vec!["result".to_string()],
                    rows: vec![vec![serde_json::json!(42)]],
                    affected_rows: None,
                    execution_time_ms: 1,
                    truncated: false,
                    warnings: vec![],
                    energy_joules: None,
                    power_watts: None,
                    device_target: None,
                    algorithm_type: None,
                    session_id: None,
                    viz_hint: None,
                })
            }
        }

        // Start shard server on random port
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let executor: Arc<dyn QueryExecutor> = Arc::new(MockQueryExecutor);

        // Spawn server task
        let server_executor = executor.clone();
        let server_handle = tokio::spawn(async move {
            let (stream, client_addr) = listener.accept().await.unwrap();
            ShardServer::handle_connection(stream, client_addr, server_executor)
                .await
                .ok();
        });

        // Give server time to start
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Connect as client and send a query
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();

        let request = QueryRequest {
            sql: "SELECT 42".to_string(),
            params: std::collections::HashMap::new(),
            args: vec![],
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };

        let payload = serde_json::to_vec(&request).unwrap();
        let msg = encode_shard_rpc(ShardRpcType::Query, &payload);
        stream.write_all(&msg).await.unwrap();

        // Read response
        let mut header = [0u8; SHARD_RPC_HEADER_SIZE];
        stream.read_exact(&mut header).await.unwrap();

        let (msg_type, payload_len) = decode_shard_rpc_header(&header).unwrap();
        assert_eq!(msg_type, ShardRpcType::Result);

        let mut response_payload = vec![0u8; payload_len];
        stream.read_exact(&mut response_payload).await.unwrap();

        let result: ShardQueryResult = serde_json::from_slice(&response_payload).unwrap();
        assert_eq!(result.columns, vec!["result"]);
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], serde_json::json!(42));

        // Cleanup
        drop(stream);
        server_handle.abort();
    }

    #[test]
    fn test_remote_shard_executor_remove_node() {
        let executor = RemoteShardExecutor::new(Duration::from_secs(5), Duration::from_secs(30));

        let addr: std::net::SocketAddr = "127.0.0.1:9001".parse().unwrap();
        executor.register_node("shard-0".to_string(), addr);
        assert_eq!(executor.list_nodes().len(), 1);

        executor.remove_node("shard-0");
        assert_eq!(executor.list_nodes().len(), 0);
    }

    #[test]
    fn test_remote_shard_executor_multiple_nodes() {
        let executor = RemoteShardExecutor::new(Duration::from_secs(5), Duration::from_secs(30));

        executor.register_node("shard-0".to_string(), "127.0.0.1:9001".parse().unwrap());
        executor.register_node("shard-1".to_string(), "127.0.0.1:9002".parse().unwrap());
        executor.register_node("shard-2".to_string(), "127.0.0.1:9003".parse().unwrap());

        let nodes = executor.list_nodes();
        assert_eq!(nodes.len(), 3);
        assert!(nodes.contains_key("shard-0"));
        assert!(nodes.contains_key("shard-1"));
        assert!(nodes.contains_key("shard-2"));
    }
}
