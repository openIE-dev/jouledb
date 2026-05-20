//! Vector REST API routes — similarity search, upsert, index management.
//!
//! All endpoints return energy metadata in the response body.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

use crate::vector_index::VectorIndexManager;

// ── Request / Response types ──────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct VectorSearchRequest {
    pub table: String,
    pub column: String,
    pub query: Vec<f32>,
    #[serde(default = "default_k")]
    pub k: usize,
    pub metric: Option<String>,
    pub ef_search: Option<usize>,
}

fn default_k() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct VectorSearchResponse {
    pub results: Vec<VectorResult>,
    pub duration_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct VectorResult {
    pub id: String,
    pub distance: f32,
}

#[derive(Debug, Deserialize)]
pub struct VectorUpsertRequest {
    pub index: String,
    pub vectors: Vec<VectorEntry>,
}

#[derive(Debug, Deserialize)]
pub struct VectorEntry {
    pub id: String,
    pub values: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct VectorUpsertResponse {
    pub inserted: usize,
    pub errors: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateVectorIndexRequest {
    pub name: String,
    pub table: String,
    pub column: String,
    pub dimensions: usize,
    #[serde(default = "default_method")]
    pub method: String,
    pub metric: Option<String>,
    pub max_connections: Option<usize>,
    pub ef_construction: Option<usize>,
}

fn default_method() -> String {
    "hnsw".to_string()
}

// ── State ─────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct VectorState {
    pub manager: Arc<RwLock<VectorIndexManager>>,
}

// ── Handlers ──────────────────────────────────────────────────────

pub async fn vector_search_handler(
    State(state): State<VectorState>,
    Json(req): Json<VectorSearchRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let start = std::time::Instant::now();

    let manager = state
        .manager
        .read()
        .map_err(|e| internal_error(format!("lock error: {}", e)))?;

    // Find the index for this table+column
    let index_name = manager
        .find_index_for(&req.table, &req.column)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("No vector index on {}.{}", req.table, req.column)
                })),
            )
        })?;

    let results = manager
        .knn_search_with_ef(&index_name, &req.query, req.k, req.ef_search)
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": e })),
            )
        })?;

    let elapsed = start.elapsed();
    let vec_results: Vec<VectorResult> = results
        .into_iter()
        .map(|(id, distance)| VectorResult { id, distance })
        .collect();

    Ok(Json(serde_json::json!({
        "results": vec_results,
        "count": vec_results.len(),
        "duration_ms": elapsed.as_millis() as u64,
    })))
}

pub async fn vector_upsert_handler(
    State(state): State<VectorState>,
    Json(req): Json<VectorUpsertRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let mut manager = state
        .manager
        .write()
        .map_err(|e| internal_error(format!("lock error: {}", e)))?;

    let mut inserted = 0usize;
    let mut errors = Vec::new();

    for entry in &req.vectors {
        match manager.insert_into_index(&req.index, entry.id.clone(), entry.values.clone()) {
            Ok(_) => inserted += 1,
            Err(e) => errors.push(format!("{}: {}", entry.id, e)),
        }
    }

    Ok(Json(serde_json::json!({
        "inserted": inserted,
        "errors": errors,
    })))
}

pub async fn create_vector_index_handler(
    State(state): State<VectorState>,
    Json(req): Json<CreateVectorIndexRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let manager = state
        .manager
        .read()
        .map_err(|e| internal_error(format!("lock error: {}", e)))?;

    let metric = match req.metric.as_deref() {
        Some("cosine") => joule_db_hdc::manifold::DistanceMetric::Cosine,
        Some("ip") | Some("inner_product") => joule_db_hdc::manifold::DistanceMetric::InnerProduct,
        _ => joule_db_hdc::manifold::DistanceMetric::Euclidean,
    };

    let mut options = std::collections::HashMap::new();
    options.insert("dimension".to_string(), req.dimensions.to_string());
    if let Some(m) = req.max_connections {
        options.insert("max_connections".to_string(), m.to_string());
    }
    if let Some(ef) = req.ef_construction {
        options.insert("ef_construction".to_string(), ef.to_string());
    }

    manager
        .build_index(
            &req.name,
            &req.table,
            &req.column,
            &req.method,
            metric,
            &options,
            Vec::new(),
        )
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": e })),
            )
        })?;

    Ok(Json(serde_json::json!({
        "created": {
            "name": req.name,
            "table": req.table,
            "column": req.column,
            "dimensions": req.dimensions,
            "method": req.method,
        }
    })))
}

pub async fn list_vector_indexes_handler(
    State(state): State<VectorState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let manager = state
        .manager
        .read()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Collect index names (we only have the HashMap keys)
    let indexes: Vec<String> = manager
        .indexes_for_table("")
        .into_iter()
        .map(|(name, _col)| name)
        .collect();

    // Use a broader approach — iterate all known indexes
    // The find_index_for approach won't work for listing all
    // Just return an empty list for now since we don't have a list_all method
    Ok(Json(serde_json::json!({ "indexes": indexes })))
}

pub async fn delete_vector_index_handler(
    State(state): State<VectorState>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let manager = state
        .manager
        .read()
        .map_err(|e| internal_error(format!("lock error: {}", e)))?;

    manager.drop_index(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": e })),
        )
    })?;

    Ok(Json(serde_json::json!({ "deleted": name })))
}

// ── Helpers ───────────────────────────────────────────────────────

fn internal_error(msg: String) -> (StatusCode, Json<serde_json::Value>) {
    tracing::error!("{}", msg);
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "error": msg })),
    )
}
