//! Database Certification, Isolation Verification & Out-of-Core Morsel OLAP HTTP Routes.
//!
//! Surfaces JouleDB's Wave 23 gap-closure engines over HTTP:
//! - `POST /api/v1/certify/hermitage_elle`: Runs Kleppmann `Hermitage` (`G0`, `G1a/b/c`, `P4`,
//!   `G2-item` Write Skew) + `Elle` `list-append` against a fresh `MvccStore` (`SSI`) and returns
//!   both the Tarjan `SCC` cycle audit and canonical Clojure EDN history.
//! - `POST /api/v1/certify/elle_analyze`: Runs Adya `ww`/`wr`/`rw` inference + Tarjan `SCC` cycle
//!   classification (`G0`..`G2`) on an arbitrary submitted transaction history.
//! - `POST /api/v1/certify/sqllogictest`: Executes portable SQLite/DuckDB `.slt` scripts directly
//!   against the live `QueryExecutor` (`statement ok/error`, `query I/R/T`, MD5 value hashing).
//! - `POST /api/v1/certify/sqlancer`: Runs `SQLancer` `TLP` (`Q(P) + Q(NOT P) + Q(P IS NULL) == Q(ALL)`)
//!   and `NoREC` differential checks against a live SQL table.
//! - `POST /api/v1/olap/morsel_join` & `POST /api/v1/olap/morsel_groupby`: Executes Radix-Partitioned
//!   Grace Out-of-Core Spill-to-Disk Hash Joins and GroupBy Aggregations under a strict
//!   `memory_budget_bytes` ceiling with 6-rail `storage_nvme_nj` receipts.

use crate::query::{QueryExecutor, QueryRequest};
use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use joule_db_core::concurrency::{
    ElleTxn, MvccTransactionManager, analyze_elle_history, run_hermitage_ssi_suite, to_elle_edn,
};
use joule_db_query::conformance_oracle::{
    run_sqlancer_tlp_and_norec_suite, run_sqllogictest_script,
};
use joule_db_query::morsel_spill::{
    ColumnMorsel, MORSEL_ROWS, SpillAggSpec, chunk_into_morsels, execute_spilling_hash_aggregate,
    execute_spilling_hash_join,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

#[derive(Clone)]
pub struct CertificationRuntime {
    pub query_executor: Arc<dyn QueryExecutor>,
}

#[derive(Debug, Default, Deserialize)]
pub struct HermitageElleReq {
    #[serde(default)]
    pub include_edn: bool,
}

#[derive(Debug, Deserialize)]
pub struct ElleAnalyzeReq {
    pub history: Vec<ElleTxn>,
    #[serde(default)]
    pub register_version_orders: BTreeMap<String, Vec<i64>>,
}

#[derive(Debug, Deserialize)]
pub struct SltRunReq {
    pub script: String,
    #[serde(default)]
    pub branch_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SqlancerRunReq {
    pub table: String,
    pub predicates: Vec<String>,
    #[serde(default)]
    pub branch_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MorselJoinReq {
    pub build_table: ColumnMorsel,
    pub probe_table: ColumnMorsel,
    pub build_key: String,
    pub probe_key: String,
    #[serde(default = "default_morsel_budget")]
    pub memory_budget_bytes: usize,
    #[serde(default = "default_partitions")]
    pub num_partitions: usize,
    #[serde(default = "default_morsel_rows")]
    pub morsel_rows: usize,
}

#[derive(Debug, Deserialize)]
pub struct MorselGroupByReq {
    pub input_table: ColumnMorsel,
    pub group_by: Vec<String>,
    pub aggregates: Vec<SpillAggSpec>,
    #[serde(default = "default_morsel_budget")]
    pub memory_budget_bytes: usize,
    #[serde(default = "default_partitions")]
    pub num_partitions: usize,
    #[serde(default = "default_morsel_rows")]
    pub morsel_rows: usize,
}

fn default_morsel_budget() -> usize {
    16 * 1024 * 1024
}

fn default_partitions() -> usize {
    16
}

fn default_morsel_rows() -> usize {
    MORSEL_ROWS
}

#[derive(Debug, Serialize)]
struct MorselApiResponse {
    result: ColumnMorsel,
    receipt: joule_db_query::morsel_spill::MorselExecutionReceipt,
}

fn bad_req(msg: impl Into<String>) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": msg.into() })),
    )
}

fn json_cell_to_slt_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "NULL".to_string(),
        serde_json::Value::Bool(b) => if *b { "1" } else { "0" }.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

async fn handle_hermitage_elle(Json(req): Json<HermitageElleReq>) -> impl IntoResponse {
    let tm = MvccTransactionManager::new();
    let mut receipt = run_hermitage_ssi_suite(&tm);
    if !req.include_edn {
        receipt.elle_edn.clear();
    }
    (StatusCode::OK, Json(serde_json::json!(receipt)))
}

async fn handle_elle_analyze(Json(req): Json<ElleAnalyzeReq>) -> impl IntoResponse {
    let report = analyze_elle_history(&req.history, &req.register_version_orders);
    let edn = to_elle_edn(&req.history);
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "report": report,
            "elle_edn": edn,
        })),
    )
}

async fn handle_sqllogictest(
    State(rt): State<CertificationRuntime>,
    Json(req): Json<SltRunReq>,
) -> impl IntoResponse {
    let branch_id = req.branch_id.clone();
    let receipt = run_sqllogictest_script(&req.script, |sql| {
        let qr = QueryRequest {
            sql: sql.to_string(),
            params: HashMap::new(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: Some(10_000),
            branch_id: branch_id.clone(),
            tenant_id: None,
        };
        let resp = rt.query_executor.execute(&qr).map_err(|e| e.message)?;
        let rows = resp
            .rows
            .iter()
            .map(|r| r.iter().map(json_cell_to_slt_string).collect())
            .collect();
        Ok(rows)
    });
    (StatusCode::OK, Json(serde_json::json!(receipt)))
}

async fn handle_sqlancer(
    State(rt): State<CertificationRuntime>,
    Json(req): Json<SqlancerRunReq>,
) -> impl IntoResponse {
    let preds: Vec<&str> = req.predicates.iter().map(String::as_str).collect();
    let branch_id = req.branch_id.clone();
    let receipt = run_sqlancer_tlp_and_norec_suite(&req.table, &preds, |sql| {
        let qr = QueryRequest {
            sql: sql.to_string(),
            params: HashMap::new(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: Some(10_000),
            branch_id: branch_id.clone(),
            tenant_id: None,
        };
        let resp = rt.query_executor.execute(&qr).map_err(|e| e.message)?;
        let first_cell = resp
            .rows
            .first()
            .and_then(|r| r.first())
            .ok_or_else(|| "empty scalar result".to_string())?;
        match first_cell {
            serde_json::Value::Number(n) => n
                .as_i64()
                .or_else(|| n.as_f64().map(|f| f as i64))
                .ok_or_else(|| "non-integer count".to_string()),
            serde_json::Value::String(s) => s
                .parse::<i64>()
                .map_err(|e| format!("parse scalar i64: {e}")),
            serde_json::Value::Null => Ok(0),
            other => Err(format!("unexpected scalar cell: {other}")),
        }
    });
    (StatusCode::OK, Json(serde_json::json!(receipt)))
}

async fn handle_morsel_join(Json(req): Json<MorselJoinReq>) -> impl IntoResponse {
    let b_morsels = match chunk_into_morsels(&req.build_table, req.morsel_rows) {
        Ok(m) => m,
        Err(e) => return bad_req(e).into_response(),
    };
    let p_morsels = match chunk_into_morsels(&req.probe_table, req.morsel_rows) {
        Ok(m) => m,
        Err(e) => return bad_req(e).into_response(),
    };
    match execute_spilling_hash_join(
        &b_morsels,
        &p_morsels,
        &req.build_key,
        &req.probe_key,
        req.memory_budget_bytes,
        req.num_partitions,
    ) {
        Ok((result, receipt)) => (
            StatusCode::OK,
            Json(serde_json::json!(MorselApiResponse { result, receipt })),
        )
            .into_response(),
        Err(e) => bad_req(e).into_response(),
    }
}

async fn handle_morsel_groupby(Json(req): Json<MorselGroupByReq>) -> impl IntoResponse {
    let morsels = match chunk_into_morsels(&req.input_table, req.morsel_rows) {
        Ok(m) => m,
        Err(e) => return bad_req(e).into_response(),
    };
    match execute_spilling_hash_aggregate(
        &morsels,
        &req.group_by,
        &req.aggregates,
        req.memory_budget_bytes,
        req.num_partitions,
    ) {
        Ok((result, receipt)) => (
            StatusCode::OK,
            Json(serde_json::json!(MorselApiResponse { result, receipt })),
        )
            .into_response(),
        Err(e) => bad_req(e).into_response(),
    }
}

pub fn certification_router(rt: CertificationRuntime) -> Router {
    Router::new()
        .route(
            "/api/v1/certify/hermitage_elle",
            post(handle_hermitage_elle),
        )
        .route("/api/v1/certify/elle_analyze", post(handle_elle_analyze))
        .route("/api/v1/certify/sqllogictest", post(handle_sqllogictest))
        .route("/api/v1/certify/sqlancer", post(handle_sqlancer))
        .route("/api/v1/olap/morsel_join", post(handle_morsel_join))
        .route("/api/v1/olap/morsel_groupby", post(handle_morsel_groupby))
        .with_state(rt)
}
