//! Feature Stores Bridge
//!
//! Integrates joule-db-features' specialized data structures (timeseries,
//! vector, fulltext, embeddings, columnar) into the server query pipeline.
//!
//! Each feature store is lazily initialized on first use and managed behind
//! `Arc<RwLock<>>` for concurrent access. SQL-style command prefixes route
//! to the appropriate store:
//!
//! - `TSWRITE` / `TSQUERY` / `TSAGGREGATE` / `TSLIST` / `TSDELETE` — timeseries
//! - `VECTOR CREATE` / `VECTOR INSERT` / `VECTOR SEARCH` / `VECTOR DELETE` — vector
//! - `FTINDEX` / `FTSEARCH` / `FTDELETE` — full-text search
//! - `EMBED` / `EMBED SEARCH` / `EMBED SIMILAR` — embeddings
//! - `COLUMNAR CREATE` / `COLUMNAR INSERT` / `COLUMNAR AGGREGATE` — columnar OLAP

use crate::lock_util::{read_lock, write_lock};
use crate::query::{QueryErrorResponse, QueryResponse};
use joule_db_features::{
    ColumnStore, ColumnStoreBuilder, DataPoint, EmbeddingConfig, EmbeddingModel, EmbeddingStore,
    FullTextConfig, FullTextIndex, SearchQuery, SimilarityMetric, TimeSeriesConfig,
    TimeSeriesStore, VectorConfig, VectorIndex,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

// ============================================================================
// Feature Store Manager
// ============================================================================

/// Centralized manager for all feature stores.
///
/// Lazily initializes stores on first access, thread-safe via RwLock.
pub struct FeatureStores {
    pub timeseries: Arc<RwLock<TimeSeriesStore>>,
    pub vectors: Arc<RwLock<HashMap<String, VectorIndex>>>,
    pub fulltext: Arc<RwLock<HashMap<String, FullTextIndex>>>,
    pub embeddings: Arc<RwLock<EmbeddingStore>>,
    pub columnar: Arc<RwLock<HashMap<String, ColumnStore>>>,
}

impl FeatureStores {
    /// Create a new feature store manager with default configs.
    pub fn new() -> Self {
        Self {
            timeseries: Arc::new(RwLock::new(TimeSeriesStore::with_defaults())),
            vectors: Arc::new(RwLock::new(HashMap::new())),
            fulltext: Arc::new(RwLock::new(HashMap::new())),
            embeddings: Arc::new(RwLock::new(EmbeddingStore::default_store())),
            columnar: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for FeatureStores {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Detection
// ============================================================================

/// Detect whether a query is a feature-store command.
pub fn is_feature_query(sql: &str) -> bool {
    let upper = sql.trim().to_uppercase();
    upper.starts_with("TSWRITE ")
        || upper.starts_with("TSQUERY ")
        || upper.starts_with("TSAGGREGATE ")
        || upper == "TSLIST"
        || upper.starts_with("TSDELETE ")
        || upper.starts_with("VECTOR ")
        || upper.starts_with("FTINDEX ")
        || upper.starts_with("FTSEARCH ")
        || upper.starts_with("FTDELETE ")
        || upper.starts_with("EMBED ")
        || upper.starts_with("COLUMNAR ")
}

// ============================================================================
// Dispatch
// ============================================================================

/// Execute a feature-store command.
pub fn execute_feature_query(
    sql: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let trimmed = sql.trim();
    let upper = trimmed.to_uppercase();

    if upper.starts_with("TSWRITE ") {
        return exec_ts_write(trimmed, stores, start);
    }
    if upper.starts_with("TSQUERY ") {
        return exec_ts_query(trimmed, stores, start);
    }
    if upper.starts_with("TSAGGREGATE ") {
        return exec_ts_aggregate(trimmed, stores, start);
    }
    if upper == "TSLIST" {
        return exec_ts_list(stores, start);
    }
    if upper.starts_with("TSDELETE ") {
        return exec_ts_delete(trimmed, stores, start);
    }
    if upper.starts_with("VECTOR ") {
        return dispatch_vector(trimmed, stores, start);
    }
    if upper.starts_with("FTINDEX ") {
        return exec_ft_index(trimmed, stores, start);
    }
    if upper.starts_with("FTSEARCH ") {
        return exec_ft_search(trimmed, stores, start);
    }
    if upper.starts_with("FTDELETE ") {
        return exec_ft_delete(trimmed, stores, start);
    }
    if upper.starts_with("EMBED ") {
        return dispatch_embed(trimmed, stores, start);
    }
    if upper.starts_with("COLUMNAR ") {
        return dispatch_columnar(trimmed, stores, start);
    }

    Err(QueryErrorResponse::syntax_error(
        &format!(
            "Unknown feature command: {}",
            trimmed.split_whitespace().next().unwrap_or("")
        ),
        1,
        1,
    ))
}

// ============================================================================
// TimeSeries
// ============================================================================

/// TSWRITE metric_name value [timestamp_ns] [tag=value ...]
fn exec_ts_write(
    sql: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let parts: Vec<&str> = sql
        .splitn(2, ' ')
        .nth(1)
        .ok_or_else(|| {
            QueryErrorResponse::syntax_error(
                "TSWRITE requires: metric value [timestamp] [tags]",
                1,
                1,
            )
        })?
        .split_whitespace()
        .collect();

    if parts.len() < 2 {
        return Err(QueryErrorResponse::syntax_error(
            "TSWRITE requires: metric value [timestamp] [tags]",
            1,
            1,
        ));
    }

    let metric = parts[0];
    let value: f64 = parts[1].parse().map_err(|_| {
        QueryErrorResponse::syntax_error(&format!("Invalid value: {}", parts[1]), 1, 1)
    })?;

    let timestamp = if parts.len() > 2 && !parts[2].contains('=') {
        parts[2].parse::<i64>().map_err(|_| {
            QueryErrorResponse::syntax_error(&format!("Invalid timestamp: {}", parts[2]), 1, 1)
        })?
    } else {
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
    };

    let mut tags = HashMap::new();
    let tag_start = if parts.len() > 2 && !parts[2].contains('=') {
        3
    } else {
        2
    };
    for &part in &parts[tag_start..] {
        if let Some((k, v)) = part.split_once('=') {
            tags.insert(k.to_string(), v.to_string());
        }
    }

    let point = DataPoint {
        timestamp,
        value,
        tags,
    };
    let ts = write_lock(&stores.timeseries);
    ts.write(metric, point);

    Ok(ok_response(start, Some(1)))
}

/// TSQUERY metric_name start_ns end_ns
fn exec_ts_query(
    sql: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let rest = sql.splitn(2, ' ').nth(1).ok_or_else(|| {
        QueryErrorResponse::syntax_error("TSQUERY requires: metric start end", 1, 1)
    })?;
    let parts: Vec<&str> = rest.split_whitespace().collect();

    if parts.len() < 3 {
        return Err(QueryErrorResponse::syntax_error(
            "TSQUERY requires: metric start end",
            1,
            1,
        ));
    }

    let metric = parts[0];
    let ts_start: i64 = parts[1].parse().map_err(|_| {
        QueryErrorResponse::syntax_error(&format!("Invalid start: {}", parts[1]), 1, 1)
    })?;
    let ts_end: i64 = parts[2].parse().map_err(|_| {
        QueryErrorResponse::syntax_error(&format!("Invalid end: {}", parts[2]), 1, 1)
    })?;

    let ts = read_lock(&stores.timeseries);
    let points = ts.query(metric, ts_start, ts_end);

    let columns = vec!["timestamp".into(), "value".into(), "tags".into()];
    let rows: Vec<Vec<serde_json::Value>> = points
        .iter()
        .map(|p| {
            vec![
                serde_json::json!(p.timestamp),
                serde_json::json!(p.value),
                serde_json::json!(p.tags),
            ]
        })
        .collect();

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("timeseries".into()),
        session_id: None,
        viz_hint: None,
    })
}

/// TSAGGREGATE metric_name start_ns end_ns interval_ms aggregation
fn exec_ts_aggregate(
    sql: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let rest = sql.splitn(2, ' ').nth(1).ok_or_else(|| {
        QueryErrorResponse::syntax_error(
            "TSAGGREGATE requires: metric start end interval aggregation",
            1,
            1,
        )
    })?;
    let parts: Vec<&str> = rest.split_whitespace().collect();

    if parts.len() < 5 {
        return Err(QueryErrorResponse::syntax_error(
            "TSAGGREGATE requires: metric start end interval_ms aggregation",
            1,
            1,
        ));
    }

    let metric = parts[0];
    let ts_start: i64 = parts[1]
        .parse()
        .map_err(|_| QueryErrorResponse::syntax_error("Invalid start timestamp", 1, 1))?;
    let ts_end: i64 = parts[2]
        .parse()
        .map_err(|_| QueryErrorResponse::syntax_error("Invalid end timestamp", 1, 1))?;
    let interval_ms: u64 = parts[3]
        .parse()
        .map_err(|_| QueryErrorResponse::syntax_error("Invalid interval", 1, 1))?;
    let aggregation = parse_ts_aggregation(parts[4])?;

    let ts = read_lock(&stores.timeseries);
    let points = ts.query_aggregate(
        metric,
        ts_start,
        ts_end,
        std::time::Duration::from_millis(interval_ms),
        aggregation,
    );

    let columns = vec!["timestamp".into(), "value".into()];
    let rows: Vec<Vec<serde_json::Value>> = points
        .iter()
        .map(|p| vec![serde_json::json!(p.timestamp), serde_json::json!(p.value)])
        .collect();

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("timeseries".into()),
        session_id: None,
        viz_hint: None,
    })
}

/// TSLIST
fn exec_ts_list(
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let ts = read_lock(&stores.timeseries);
    let metrics = ts.list_metrics();

    let columns = vec!["metric".into()];
    let rows: Vec<Vec<serde_json::Value>> = metrics
        .into_iter()
        .map(|m| vec![serde_json::json!(m)])
        .collect();

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("timeseries".into()),
        session_id: None,
        viz_hint: None,
    })
}

/// TSDELETE metric_name
fn exec_ts_delete(
    sql: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let metric = sql
        .splitn(2, ' ')
        .nth(1)
        .ok_or_else(|| QueryErrorResponse::syntax_error("TSDELETE requires: metric", 1, 1))?
        .trim();

    let ts = write_lock(&stores.timeseries);
    let deleted = ts.delete_metric(metric);

    Ok(ok_response(start, if deleted { Some(1) } else { Some(0) }))
}

fn parse_ts_aggregation(s: &str) -> Result<joule_db_features::Aggregation, QueryErrorResponse> {
    match s.to_uppercase().as_str() {
        "SUM" => Ok(joule_db_features::Aggregation::Sum),
        "COUNT" => Ok(joule_db_features::Aggregation::Count),
        "MEAN" | "AVG" => Ok(joule_db_features::Aggregation::Mean),
        "MIN" => Ok(joule_db_features::Aggregation::Min),
        "MAX" => Ok(joule_db_features::Aggregation::Max),
        "FIRST" => Ok(joule_db_features::Aggregation::First),
        "LAST" => Ok(joule_db_features::Aggregation::Last),
        "STDDEV" => Ok(joule_db_features::Aggregation::Stddev),
        "VARIANCE" => Ok(joule_db_features::Aggregation::Variance),
        _ => Err(QueryErrorResponse::syntax_error(
            &format!(
                "Unknown aggregation: {}. Use SUM/COUNT/MEAN/MIN/MAX/FIRST/LAST/STDDEV/VARIANCE",
                s
            ),
            1,
            1,
        )),
    }
}

// ============================================================================
// Vector
// ============================================================================

/// Dispatch VECTOR sub-commands.
fn dispatch_vector(
    sql: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let rest = sql.splitn(2, ' ').nth(1).ok_or_else(|| {
        QueryErrorResponse::syntax_error(
            "VECTOR requires a sub-command: CREATE/INSERT/SEARCH/DELETE/LIST",
            1,
            1,
        )
    })?;
    let upper = rest.trim().to_uppercase();

    if upper.starts_with("CREATE ") {
        return exec_vector_create(rest.trim(), stores, start);
    }
    if upper.starts_with("INSERT ") {
        return exec_vector_insert(rest.trim(), stores, start);
    }
    if upper.starts_with("SEARCH ") {
        return exec_vector_search(rest.trim(), stores, start);
    }
    if upper.starts_with("DELETE ") {
        return exec_vector_delete(rest.trim(), stores, start);
    }
    if upper == "LIST" {
        return exec_vector_list(stores, start);
    }

    Err(QueryErrorResponse::syntax_error(
        "VECTOR sub-commands: CREATE index dims [metric], INSERT index id [v1,v2,...], SEARCH index [v1,v2,...] k, DELETE index id, LIST",
        1,
        1,
    ))
}

/// VECTOR CREATE index_name dimensions [metric]
fn exec_vector_create(
    rest: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let parts: Vec<&str> = rest
        .splitn(2, ' ')
        .nth(1)
        .ok_or_else(|| {
            QueryErrorResponse::syntax_error(
                "VECTOR CREATE requires: name dimensions [metric]",
                1,
                1,
            )
        })?
        .split_whitespace()
        .collect();

    if parts.is_empty() {
        return Err(QueryErrorResponse::syntax_error(
            "VECTOR CREATE requires: name dimensions [metric]",
            1,
            1,
        ));
    }

    let name = parts[0];
    let dims: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let metric = match parts.get(2).map(|s| s.to_uppercase()).as_deref() {
        Some("EUCLIDEAN") => SimilarityMetric::Euclidean,
        Some("DOT") | Some("DOTPRODUCT") => SimilarityMetric::DotProduct,
        Some("MANHATTAN") => SimilarityMetric::Manhattan,
        _ => SimilarityMetric::Cosine,
    };

    let config = VectorConfig {
        dimensions: dims,
        metric,
        ..VectorConfig::default()
    };

    let mut vectors = write_lock(&stores.vectors);
    vectors.insert(name.to_string(), VectorIndex::new(config));

    Ok(ok_response(start, None))
}

/// VECTOR INSERT index_name id [v1,v2,v3,...]
fn exec_vector_insert(
    rest: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let parts: Vec<&str> = rest
        .splitn(2, ' ')
        .nth(1)
        .ok_or_else(|| {
            QueryErrorResponse::syntax_error("VECTOR INSERT requires: index id [v1,v2,...]", 1, 1)
        })?
        .splitn(3, ' ')
        .collect();

    if parts.len() < 3 {
        return Err(QueryErrorResponse::syntax_error(
            "VECTOR INSERT requires: index id [v1,v2,...]",
            1,
            1,
        ));
    }

    let index_name = parts[0];
    let id = parts[1];
    let vector = parse_vector(parts[2])?;

    let mut vectors = write_lock(&stores.vectors);
    let index = vectors.get_mut(index_name).ok_or_else(|| {
        QueryErrorResponse::execution_error(&format!(
            "Vector index '{}' not found. Use VECTOR CREATE first.",
            index_name
        ))
    })?;

    index
        .insert(id, vector)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("Vector insert error: {}", e)))?;

    Ok(ok_response(start, Some(1)))
}

/// VECTOR SEARCH index_name [v1,v2,...] k
fn exec_vector_search(
    rest: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let parts: Vec<&str> = rest
        .splitn(2, ' ')
        .nth(1)
        .ok_or_else(|| {
            QueryErrorResponse::syntax_error("VECTOR SEARCH requires: index [v1,v2,...] k", 1, 1)
        })?
        .splitn(3, ' ')
        .collect();

    if parts.len() < 3 {
        return Err(QueryErrorResponse::syntax_error(
            "VECTOR SEARCH requires: index [v1,v2,...] k",
            1,
            1,
        ));
    }

    let index_name = parts[0];
    let query = parse_vector(parts[1])?;
    let k: usize = parts[2]
        .parse()
        .map_err(|_| QueryErrorResponse::syntax_error(&format!("Invalid k: {}", parts[2]), 1, 1))?;

    let vectors = read_lock(&stores.vectors);
    let index = vectors.get(index_name).ok_or_else(|| {
        QueryErrorResponse::execution_error(&format!("Vector index '{}' not found", index_name))
    })?;

    let results = index
        .search(&query, k)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("Vector search error: {}", e)))?;

    let columns = vec!["id".into(), "score".into(), "metadata".into()];
    let rows: Vec<Vec<serde_json::Value>> = results
        .iter()
        .map(|r| {
            vec![
                serde_json::json!(r.id),
                serde_json::json!(r.score),
                r.metadata.clone().unwrap_or(serde_json::Value::Null),
            ]
        })
        .collect();

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("vector".into()),
        session_id: None,
        viz_hint: None,
    })
}

/// VECTOR DELETE index_name id
fn exec_vector_delete(
    rest: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let parts: Vec<&str> = rest
        .splitn(2, ' ')
        .nth(1)
        .ok_or_else(|| QueryErrorResponse::syntax_error("VECTOR DELETE requires: index id", 1, 1))?
        .split_whitespace()
        .collect();

    if parts.len() < 2 {
        return Err(QueryErrorResponse::syntax_error(
            "VECTOR DELETE requires: index id",
            1,
            1,
        ));
    }

    let mut vectors = write_lock(&stores.vectors);
    let index = vectors.get_mut(parts[0]).ok_or_else(|| {
        QueryErrorResponse::execution_error(&format!("Vector index '{}' not found", parts[0]))
    })?;

    let removed = index.remove(parts[1]);

    Ok(ok_response(start, if removed { Some(1) } else { Some(0) }))
}

/// VECTOR LIST
fn exec_vector_list(
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let vectors = read_lock(&stores.vectors);
    let columns = vec![
        "index_name".into(),
        "dimensions".into(),
        "metric".into(),
        "count".into(),
    ];
    let rows: Vec<Vec<serde_json::Value>> = vectors
        .iter()
        .map(|(name, idx)| {
            vec![
                serde_json::json!(name),
                serde_json::json!(idx.config().dimensions),
                serde_json::json!(format!("{:?}", idx.config().metric)),
                serde_json::json!(idx.len()),
            ]
        })
        .collect();

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("vector".into()),
        session_id: None,
        viz_hint: None,
    })
}

fn parse_vector(s: &str) -> Result<Vec<f32>, QueryErrorResponse> {
    let trimmed = s.trim().trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(',')
        .map(|v| {
            v.trim().parse::<f32>().map_err(|_| {
                QueryErrorResponse::syntax_error(
                    &format!("Invalid vector component: {}", v.trim()),
                    1,
                    1,
                )
            })
        })
        .collect()
}

// ============================================================================
// Full-Text Search
// ============================================================================

/// FTINDEX index_name doc_id content...
fn exec_ft_index(
    sql: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let rest = sql.splitn(2, ' ').nth(1).ok_or_else(|| {
        QueryErrorResponse::syntax_error("FTINDEX requires: index doc_id content", 1, 1)
    })?;
    let parts: Vec<&str> = rest.splitn(3, ' ').collect();

    if parts.len() < 3 {
        return Err(QueryErrorResponse::syntax_error(
            "FTINDEX requires: index doc_id content",
            1,
            1,
        ));
    }

    let index_name = parts[0];
    let doc_id = parts[1];
    let content = parts[2];

    let mut indices = write_lock(&stores.fulltext);
    let index = indices
        .entry(index_name.to_string())
        .or_insert_with(FullTextIndex::default_index);

    index.add_document(doc_id, content).map_err(|e| {
        QueryErrorResponse::execution_error(&format!("Fulltext index error: {}", e))
    })?;

    Ok(ok_response(start, Some(1)))
}

/// FTSEARCH index_name query_text [LIMIT n]
fn exec_ft_search(
    sql: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let rest = sql.splitn(2, ' ').nth(1).ok_or_else(|| {
        QueryErrorResponse::syntax_error("FTSEARCH requires: index query [LIMIT n]", 1, 1)
    })?;

    // Split off index name, then parse remaining for query and LIMIT
    let parts: Vec<&str> = rest.splitn(2, ' ').collect();
    if parts.len() < 2 {
        return Err(QueryErrorResponse::syntax_error(
            "FTSEARCH requires: index query [LIMIT n]",
            1,
            1,
        ));
    }

    let index_name = parts[0];
    let query_and_limit = parts[1];

    // Check for LIMIT clause
    let (query_text, limit) = if let Some(pos) = query_and_limit.to_uppercase().rfind(" LIMIT ") {
        let q = &query_and_limit[..pos];
        let l: usize = query_and_limit[pos + 7..].trim().parse().unwrap_or(100);
        (q, l)
    } else {
        (query_and_limit, 100)
    };

    let query = SearchQuery::parse(query_text);

    let indices = read_lock(&stores.fulltext);
    let index = indices.get(index_name).ok_or_else(|| {
        QueryErrorResponse::execution_error(&format!("Fulltext index '{}' not found", index_name))
    })?;

    let hits = index.search_with_limit(query, limit);

    let columns = vec!["doc_id".into(), "score".into(), "highlights".into()];
    let rows: Vec<Vec<serde_json::Value>> = hits
        .iter()
        .map(|h| {
            let highlights: Vec<String> = h
                .highlights
                .iter()
                .map(|hl| format!("{}@{:?}", hl.term, hl.positions))
                .collect();
            vec![
                serde_json::json!(h.id),
                serde_json::json!(h.score),
                serde_json::json!(highlights),
            ]
        })
        .collect();

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("fulltext".into()),
        session_id: None,
        viz_hint: None,
    })
}

/// FTDELETE index_name doc_id
fn exec_ft_delete(
    sql: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let rest = sql
        .splitn(2, ' ')
        .nth(1)
        .ok_or_else(|| QueryErrorResponse::syntax_error("FTDELETE requires: index doc_id", 1, 1))?;
    let parts: Vec<&str> = rest.split_whitespace().collect();

    if parts.len() < 2 {
        return Err(QueryErrorResponse::syntax_error(
            "FTDELETE requires: index doc_id",
            1,
            1,
        ));
    }

    let mut indices = write_lock(&stores.fulltext);
    let index = indices.get_mut(parts[0]).ok_or_else(|| {
        QueryErrorResponse::execution_error(&format!("Fulltext index '{}' not found", parts[0]))
    })?;

    let removed = index.remove(parts[1]);

    Ok(ok_response(start, if removed { Some(1) } else { Some(0) }))
}

// ============================================================================
// Embeddings
// ============================================================================

/// Dispatch EMBED sub-commands.
fn dispatch_embed(
    sql: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let rest = sql.splitn(2, ' ').nth(1).ok_or_else(|| {
        QueryErrorResponse::syntax_error("EMBED requires: TEXT/SEARCH/SIMILAR", 1, 1)
    })?;
    let upper = rest.trim().to_uppercase();

    if upper.starts_with("TEXT ") {
        return exec_embed_text(rest.trim(), stores, start);
    }
    if upper.starts_with("SIMILAR ") {
        return exec_embed_similar(rest.trim(), stores, start);
    }
    if upper.starts_with("SEARCH ") {
        return exec_embed_search(rest.trim(), stores, start);
    }

    Err(QueryErrorResponse::syntax_error(
        "EMBED sub-commands: TEXT id content, SIMILAR id k, SEARCH [v1,v2,...] k",
        1,
        1,
    ))
}

/// EMBED TEXT id content...
fn exec_embed_text(
    rest: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let parts: Vec<&str> = rest
        .splitn(2, ' ')
        .nth(1)
        .ok_or_else(|| QueryErrorResponse::syntax_error("EMBED TEXT requires: id content", 1, 1))?
        .splitn(2, ' ')
        .collect();

    if parts.len() < 2 {
        return Err(QueryErrorResponse::syntax_error(
            "EMBED TEXT requires: id content",
            1,
            1,
        ));
    }

    let id = parts[0];
    let text = parts[1];

    let mut embed = write_lock(&stores.embeddings);
    embed
        .embed_text(id, text)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("Embedding error: {}", e)))?;

    Ok(ok_response(start, Some(1)))
}

/// EMBED SIMILAR id k
fn exec_embed_similar(
    rest: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let parts: Vec<&str> = rest
        .splitn(2, ' ')
        .nth(1)
        .ok_or_else(|| QueryErrorResponse::syntax_error("EMBED SIMILAR requires: id k", 1, 1))?
        .split_whitespace()
        .collect();

    if parts.len() < 2 {
        return Err(QueryErrorResponse::syntax_error(
            "EMBED SIMILAR requires: id k",
            1,
            1,
        ));
    }

    let id = parts[0];
    let k: usize = parts[1]
        .parse()
        .map_err(|_| QueryErrorResponse::syntax_error(&format!("Invalid k: {}", parts[1]), 1, 1))?;

    let embed = read_lock(&stores.embeddings);
    let results = embed
        .find_similar(id, k)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("Embedding error: {}", e)))?;

    let columns = vec!["id".into(), "similarity".into(), "text".into()];
    let rows: Vec<Vec<serde_json::Value>> = results
        .iter()
        .map(|r| {
            vec![
                serde_json::json!(r.id),
                serde_json::json!(r.similarity),
                serde_json::json!(r.text),
            ]
        })
        .collect();

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("embeddings".into()),
        session_id: None,
        viz_hint: None,
    })
}

/// EMBED SEARCH [v1,v2,...] k
fn exec_embed_search(
    rest: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let parts: Vec<&str> = rest
        .splitn(2, ' ')
        .nth(1)
        .ok_or_else(|| {
            QueryErrorResponse::syntax_error("EMBED SEARCH requires: [v1,v2,...] k", 1, 1)
        })?
        .splitn(2, ' ')
        .collect();

    if parts.len() < 2 {
        return Err(QueryErrorResponse::syntax_error(
            "EMBED SEARCH requires: [v1,v2,...] k",
            1,
            1,
        ));
    }

    let query = parse_vector(parts[0])?;
    let k: usize = parts[1]
        .parse()
        .map_err(|_| QueryErrorResponse::syntax_error(&format!("Invalid k: {}", parts[1]), 1, 1))?;

    let embed = read_lock(&stores.embeddings);
    let results = embed.find_similar_to_vector(&query, k, None).map_err(|e| {
        QueryErrorResponse::execution_error(&format!("Embedding search error: {}", e))
    })?;

    let columns = vec!["id".into(), "similarity".into(), "text".into()];
    let rows: Vec<Vec<serde_json::Value>> = results
        .iter()
        .map(|r| {
            vec![
                serde_json::json!(r.id),
                serde_json::json!(r.similarity),
                serde_json::json!(r.text),
            ]
        })
        .collect();

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("embeddings".into()),
        session_id: None,
        viz_hint: None,
    })
}

// ============================================================================
// Columnar
// ============================================================================

/// Dispatch COLUMNAR sub-commands.
fn dispatch_columnar(
    sql: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let rest = sql.splitn(2, ' ').nth(1).ok_or_else(|| {
        QueryErrorResponse::syntax_error(
            "COLUMNAR requires a sub-command: CREATE/INSERT/AGGREGATE/LIST",
            1,
            1,
        )
    })?;
    let upper = rest.trim().to_uppercase();

    if upper.starts_with("CREATE ") {
        return exec_columnar_create(rest.trim(), stores, start);
    }
    if upper.starts_with("INSERT ") {
        return exec_columnar_insert(rest.trim(), stores, start);
    }
    if upper.starts_with("AGGREGATE ") {
        return exec_columnar_aggregate(rest.trim(), stores, start);
    }
    if upper == "LIST" {
        return exec_columnar_list(stores, start);
    }

    Err(QueryErrorResponse::syntax_error(
        "COLUMNAR sub-commands: CREATE name col:type..., INSERT name val..., AGGREGATE name col agg, LIST",
        1,
        1,
    ))
}

/// COLUMNAR CREATE store_name col1:type1 col2:type2 ...
fn exec_columnar_create(
    rest: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let parts: Vec<&str> = rest
        .splitn(2, ' ')
        .nth(1)
        .ok_or_else(|| {
            QueryErrorResponse::syntax_error("COLUMNAR CREATE requires: name col:type ...", 1, 1)
        })?
        .split_whitespace()
        .collect();

    if parts.is_empty() {
        return Err(QueryErrorResponse::syntax_error(
            "COLUMNAR CREATE requires: name col:type ...",
            1,
            1,
        ));
    }

    let name = parts[0];
    let mut store = ColumnStore::new(name);

    for &col_def in &parts[1..] {
        if let Some((col_name, type_str)) = col_def.split_once(':') {
            let dt = parse_columnar_type(type_str)?;
            store.add_column(col_name, dt);
        } else {
            return Err(QueryErrorResponse::syntax_error(
                &format!("Column definition must be name:type, got '{}'", col_def),
                1,
                1,
            ));
        }
    }

    let mut cols = write_lock(&stores.columnar);
    cols.insert(name.to_string(), store);

    Ok(ok_response(start, None))
}

/// COLUMNAR INSERT store_name val1 val2 ...
fn exec_columnar_insert(
    rest: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let parts: Vec<&str> = rest
        .splitn(2, ' ')
        .nth(1)
        .ok_or_else(|| {
            QueryErrorResponse::syntax_error("COLUMNAR INSERT requires: name val1 val2 ...", 1, 1)
        })?
        .splitn(2, ' ')
        .collect();

    if parts.len() < 2 {
        return Err(QueryErrorResponse::syntax_error(
            "COLUMNAR INSERT requires: name val1 val2 ...",
            1,
            1,
        ));
    }

    let name = parts[0];
    let values_str: Vec<&str> = parts[1].split_whitespace().collect();

    let values: Vec<joule_db_features::columnar::Value> =
        values_str.iter().map(|v| parse_columnar_value(v)).collect();

    let mut cols = write_lock(&stores.columnar);
    let store = cols.get_mut(name).ok_or_else(|| {
        QueryErrorResponse::execution_error(&format!("Columnar store '{}' not found", name))
    })?;

    store.insert_row(values).map_err(|e| {
        QueryErrorResponse::execution_error(&format!("Columnar insert error: {}", e))
    })?;

    Ok(ok_response(start, Some(1)))
}

/// COLUMNAR AGGREGATE store_name column aggregation
fn exec_columnar_aggregate(
    rest: &str,
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let parts: Vec<&str> = rest
        .splitn(2, ' ')
        .nth(1)
        .ok_or_else(|| {
            QueryErrorResponse::syntax_error(
                "COLUMNAR AGGREGATE requires: name column aggregation",
                1,
                1,
            )
        })?
        .split_whitespace()
        .collect();

    if parts.len() < 3 {
        return Err(QueryErrorResponse::syntax_error(
            "COLUMNAR AGGREGATE requires: name column aggregation",
            1,
            1,
        ));
    }

    let name = parts[0];
    let column = parts[1];
    let agg = parse_columnar_aggregation(parts[2])?;

    let cols = read_lock(&stores.columnar);
    let store = cols.get(name).ok_or_else(|| {
        QueryErrorResponse::execution_error(&format!("Columnar store '{}' not found", name))
    })?;

    let result = store.aggregate(column, agg);

    let columns = vec!["column".into(), "aggregation".into(), "result".into()];
    let value = match result {
        Some(v) => columnar_value_to_json(&v),
        None => serde_json::Value::Null,
    };
    let rows = vec![vec![
        serde_json::json!(column),
        serde_json::json!(parts[2].to_uppercase()),
        value,
    ]];

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("columnar".into()),
        session_id: None,
        viz_hint: None,
    })
}

/// COLUMNAR LIST
fn exec_columnar_list(
    stores: &FeatureStores,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let cols = read_lock(&stores.columnar);
    let columns = vec!["store_name".into(), "columns".into(), "rows".into()];
    let rows: Vec<Vec<serde_json::Value>> = cols
        .iter()
        .map(|(name, store)| {
            vec![
                serde_json::json!(name),
                serde_json::json!(store.column_names()),
                serde_json::json!(store.row_count()),
            ]
        })
        .collect();

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("columnar".into()),
        session_id: None,
        viz_hint: None,
    })
}

fn parse_columnar_type(s: &str) -> Result<joule_db_features::DataType, QueryErrorResponse> {
    match s.to_uppercase().as_str() {
        "INT" | "INT64" | "INTEGER" => Ok(joule_db_features::DataType::Int64),
        "INT32" => Ok(joule_db_features::DataType::Int32),
        "FLOAT" | "FLOAT64" | "DOUBLE" => Ok(joule_db_features::DataType::Float64),
        "FLOAT32" | "REAL" => Ok(joule_db_features::DataType::Float32),
        "STRING" | "TEXT" | "VARCHAR" => Ok(joule_db_features::DataType::String),
        "BOOL" | "BOOLEAN" => Ok(joule_db_features::DataType::Boolean),
        "TIMESTAMP" => Ok(joule_db_features::DataType::Timestamp),
        _ => Err(QueryErrorResponse::syntax_error(
            &format!(
                "Unknown columnar type: {}. Use INT/FLOAT/STRING/BOOL/TIMESTAMP",
                s
            ),
            1,
            1,
        )),
    }
}

fn parse_columnar_aggregation(
    s: &str,
) -> Result<joule_db_features::ColumnarAggregation, QueryErrorResponse> {
    match s.to_uppercase().as_str() {
        "COUNT" => Ok(joule_db_features::ColumnarAggregation::Count),
        "SUM" => Ok(joule_db_features::ColumnarAggregation::Sum),
        "MIN" => Ok(joule_db_features::ColumnarAggregation::Min),
        "MAX" => Ok(joule_db_features::ColumnarAggregation::Max),
        "AVG" | "MEAN" => Ok(joule_db_features::ColumnarAggregation::Avg),
        "FIRST" => Ok(joule_db_features::ColumnarAggregation::First),
        "LAST" => Ok(joule_db_features::ColumnarAggregation::Last),
        _ => Err(QueryErrorResponse::syntax_error(
            &format!(
                "Unknown aggregation: {}. Use COUNT/SUM/MIN/MAX/AVG/FIRST/LAST",
                s
            ),
            1,
            1,
        )),
    }
}

fn parse_columnar_value(s: &str) -> joule_db_features::columnar::Value {
    if s.eq_ignore_ascii_case("null") {
        return joule_db_features::columnar::Value::Null;
    }
    if s.eq_ignore_ascii_case("true") {
        return joule_db_features::columnar::Value::Boolean(true);
    }
    if s.eq_ignore_ascii_case("false") {
        return joule_db_features::columnar::Value::Boolean(false);
    }
    if let Ok(i) = s.parse::<i64>() {
        return joule_db_features::columnar::Value::Int64(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return joule_db_features::columnar::Value::Float64(f);
    }
    // Strip surrounding quotes if present
    let text = s.trim_matches('\'').trim_matches('"');
    joule_db_features::columnar::Value::String(text.to_string())
}

fn columnar_value_to_json(v: &joule_db_features::columnar::Value) -> serde_json::Value {
    match v {
        joule_db_features::columnar::Value::Null => serde_json::Value::Null,
        joule_db_features::columnar::Value::Boolean(b) => serde_json::json!(b),
        joule_db_features::columnar::Value::Int8(i) => serde_json::json!(i),
        joule_db_features::columnar::Value::Int16(i) => serde_json::json!(i),
        joule_db_features::columnar::Value::Int32(i) => serde_json::json!(i),
        joule_db_features::columnar::Value::Int64(i) => serde_json::json!(i),
        joule_db_features::columnar::Value::Float32(f) => serde_json::json!(f),
        joule_db_features::columnar::Value::Float64(f) => serde_json::json!(f),
        joule_db_features::columnar::Value::String(s) => serde_json::json!(s),
        joule_db_features::columnar::Value::Binary(b) => {
            serde_json::json!(format!("<{} bytes>", b.len()))
        }
        joule_db_features::columnar::Value::Timestamp(t) => serde_json::json!(t),
        joule_db_features::columnar::Value::Date(d) => serde_json::json!(d),
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn ok_response(start: Instant, affected: Option<usize>) -> QueryResponse {
    QueryResponse {
        columns: vec!["result".into()],
        rows: vec![vec![serde_json::json!("OK")]],
        affected_rows: affected,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: None,
        session_id: None,
        viz_hint: None,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_stores() -> FeatureStores {
        FeatureStores::new()
    }

    // -- Detection --

    #[test]
    fn test_is_feature_query_positive() {
        assert!(is_feature_query("TSWRITE cpu.load 42.5"));
        assert!(is_feature_query("TSQUERY cpu.load 0 999999"));
        assert!(is_feature_query("TSAGGREGATE cpu.load 0 999 1000 AVG"));
        assert!(is_feature_query("TSLIST"));
        assert!(is_feature_query("TSDELETE cpu.load"));
        assert!(is_feature_query("VECTOR CREATE my_idx 128"));
        assert!(is_feature_query("VECTOR SEARCH my_idx [1,2,3] 5"));
        assert!(is_feature_query("FTINDEX docs d1 hello world"));
        assert!(is_feature_query("FTSEARCH docs hello"));
        assert!(is_feature_query("EMBED TEXT d1 hello world"));
        assert!(is_feature_query("COLUMNAR CREATE sales amount:float"));
    }

    #[test]
    fn test_is_feature_query_negative() {
        assert!(!is_feature_query("SELECT * FROM users"));
        assert!(!is_feature_query("INSERT INTO users VALUES (1)"));
        assert!(!is_feature_query("MATCH (n) RETURN n"));
        assert!(!is_feature_query("FROM sensor TRANSFORM fft()"));
    }

    // -- TimeSeries --

    #[test]
    fn test_ts_write_and_query() {
        let stores = test_stores();
        let start = Instant::now();

        // Write a point
        let res = execute_feature_query("TSWRITE cpu.load 42.5 1000000", &stores, start).unwrap();
        assert_eq!(res.affected_rows, Some(1));

        // Query it back
        let res = execute_feature_query("TSQUERY cpu.load 0 9999999", &stores, start).unwrap();
        assert_eq!(res.columns, vec!["timestamp", "value", "tags"]);
        assert_eq!(res.rows.len(), 1);
        assert_eq!(res.rows[0][1], serde_json::json!(42.5));
    }

    #[test]
    fn test_ts_list_and_delete() {
        let stores = test_stores();
        let start = Instant::now();

        execute_feature_query("TSWRITE cpu.load 1.0", &stores, start).unwrap();
        execute_feature_query("TSWRITE mem.used 80.0", &stores, start).unwrap();

        let res = execute_feature_query("TSLIST", &stores, start).unwrap();
        assert_eq!(res.rows.len(), 2);

        execute_feature_query("TSDELETE cpu.load", &stores, start).unwrap();

        let res = execute_feature_query("TSLIST", &stores, start).unwrap();
        assert_eq!(res.rows.len(), 1);
    }

    #[test]
    fn test_ts_aggregate() {
        let stores = test_stores();
        let start = Instant::now();

        for i in 0..10 {
            execute_feature_query(
                &format!("TSWRITE temp {} {}", i as f64 * 10.0, i * 1_000_000),
                &stores,
                start,
            )
            .unwrap();
        }

        let res = execute_feature_query("TSAGGREGATE temp 0 10000000 5000000 AVG", &stores, start)
            .unwrap();
        assert!(!res.rows.is_empty());
        assert_eq!(res.algorithm_type, Some("timeseries".into()));
    }

    // -- Vector --

    #[test]
    fn test_vector_create_insert_search() {
        let stores = test_stores();
        let start = Instant::now();

        execute_feature_query("VECTOR CREATE idx 3 cosine", &stores, start).unwrap();
        execute_feature_query("VECTOR INSERT idx v1 [1.0,0.0,0.0]", &stores, start).unwrap();
        execute_feature_query("VECTOR INSERT idx v2 [0.0,1.0,0.0]", &stores, start).unwrap();
        execute_feature_query("VECTOR INSERT idx v3 [1.0,0.1,0.0]", &stores, start).unwrap();

        let res =
            execute_feature_query("VECTOR SEARCH idx [1.0,0.0,0.0] 2", &stores, start).unwrap();
        assert_eq!(res.columns, vec!["id", "score", "metadata"]);
        assert!(res.rows.len() <= 2);
        // v1 should be the closest match
        assert_eq!(res.rows[0][0], serde_json::json!("v1"));
    }

    #[test]
    fn test_vector_list_and_delete() {
        let stores = test_stores();
        let start = Instant::now();

        execute_feature_query("VECTOR CREATE idx1 4", &stores, start).unwrap();
        execute_feature_query("VECTOR CREATE idx2 8 euclidean", &stores, start).unwrap();

        let res = execute_feature_query("VECTOR LIST", &stores, start).unwrap();
        assert_eq!(res.rows.len(), 2);

        execute_feature_query("VECTOR INSERT idx1 v1 [1,2,3,4]", &stores, start).unwrap();
        let res = execute_feature_query("VECTOR DELETE idx1 v1", &stores, start).unwrap();
        assert_eq!(res.affected_rows, Some(1));
    }

    // -- Full-Text --

    #[test]
    fn test_ft_index_and_search() {
        let stores = test_stores();
        let start = Instant::now();

        execute_feature_query(
            "FTINDEX docs d1 The quick brown fox jumps over the lazy dog",
            &stores,
            start,
        )
        .unwrap();
        execute_feature_query(
            "FTINDEX docs d2 A fast red car drives past the sleeping cat",
            &stores,
            start,
        )
        .unwrap();

        let res = execute_feature_query("FTSEARCH docs quick", &stores, start).unwrap();
        assert_eq!(res.columns, vec!["doc_id", "score", "highlights"]);
        assert!(!res.rows.is_empty());
        assert_eq!(res.rows[0][0], serde_json::json!("d1"));
    }

    #[test]
    fn test_ft_delete() {
        let stores = test_stores();
        let start = Instant::now();

        execute_feature_query("FTINDEX idx d1 hello world", &stores, start).unwrap();
        let res = execute_feature_query("FTDELETE idx d1", &stores, start).unwrap();
        assert_eq!(res.affected_rows, Some(1));
    }

    // -- Embeddings --

    #[test]
    fn test_embed_text_and_similar() {
        let stores = test_stores();
        let start = Instant::now();

        execute_feature_query("EMBED TEXT d1 machine learning algorithms", &stores, start).unwrap();
        execute_feature_query("EMBED TEXT d2 deep neural networks", &stores, start).unwrap();
        execute_feature_query("EMBED TEXT d3 cooking recipes for dinner", &stores, start).unwrap();

        let res = execute_feature_query("EMBED SIMILAR d1 2", &stores, start).unwrap();
        assert_eq!(res.columns, vec!["id", "similarity", "text"]);
        // Should find d2 or d3 as similar (but not d1 itself)
        assert!(res.rows.len() <= 2);
    }

    // -- Columnar --

    #[test]
    fn test_columnar_create_insert_aggregate() {
        let stores = test_stores();
        let start = Instant::now();

        execute_feature_query(
            "COLUMNAR CREATE sales amount:float quantity:int",
            &stores,
            start,
        )
        .unwrap();
        execute_feature_query("COLUMNAR INSERT sales 99.99 5", &stores, start).unwrap();
        execute_feature_query("COLUMNAR INSERT sales 49.99 3", &stores, start).unwrap();
        execute_feature_query("COLUMNAR INSERT sales 149.99 1", &stores, start).unwrap();

        let res =
            execute_feature_query("COLUMNAR AGGREGATE sales amount SUM", &stores, start).unwrap();
        assert_eq!(res.columns, vec!["column", "aggregation", "result"]);
        assert_eq!(res.rows[0][1], serde_json::json!("SUM"));
        // Sum should be 299.97
        let sum = res.rows[0][2].as_f64().unwrap();
        assert!((sum - 299.97).abs() < 0.01);
    }

    #[test]
    fn test_columnar_list() {
        let stores = test_stores();
        let start = Instant::now();

        execute_feature_query("COLUMNAR CREATE store1 a:int b:float", &stores, start).unwrap();
        execute_feature_query("COLUMNAR CREATE store2 x:string", &stores, start).unwrap();

        let res = execute_feature_query("COLUMNAR LIST", &stores, start).unwrap();
        assert_eq!(res.rows.len(), 2);
    }

    // -- Error handling --

    #[test]
    fn test_vector_not_found() {
        let stores = test_stores();
        let start = Instant::now();
        let res = execute_feature_query("VECTOR SEARCH noexist [1,2] 5", &stores, start);
        assert!(res.is_err());
    }

    #[test]
    fn test_parse_vector() {
        let v = parse_vector("[1.0,2.5,3.0]").unwrap();
        assert_eq!(v, vec![1.0, 2.5, 3.0]);

        let v = parse_vector("4,5,6").unwrap();
        assert_eq!(v, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_ts_write_with_tags() {
        let stores = test_stores();
        let start = Instant::now();

        let res = execute_feature_query(
            "TSWRITE cpu.load 75.0 1000 host=server1 dc=us-east",
            &stores,
            start,
        )
        .unwrap();
        assert_eq!(res.affected_rows, Some(1));

        let res = execute_feature_query("TSQUERY cpu.load 0 999999999", &stores, start).unwrap();
        assert_eq!(res.rows.len(), 1);
        let tags = &res.rows[0][2];
        assert_eq!(tags["host"], "server1");
        assert_eq!(tags["dc"], "us-east");
    }
}
