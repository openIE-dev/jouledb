#![cfg(feature = "http")]

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{Json, Router, routing::get};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::batch::{BatchCommitment, ReceiptBatch};
use crate::committer::ReceiptStore;
use crate::merkle::MerkleProof;
use crate::receipt::LedgerEnergyReceipt;

/// Shared state for verification handlers.
#[derive(Clone)]
pub struct LedgerVerifyState {
    pub store: Arc<RwLock<ReceiptStore>>,
}

/// Response for `GET /api/v1/ledger/receipts/{receipt_id}/verify`.
#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    pub receipt: LedgerEnergyReceipt,
    pub proof: MerkleProof,
    pub batch: BatchSummary,
    pub commitment: BatchCommitment,
    pub verified: bool,
}

/// Summary of a batch (used in list and detail views).
#[derive(Debug, Clone, Serialize)]
pub struct BatchSummary {
    pub batch_id: String,
    pub merkle_root: String,
    pub receipt_count: usize,
    pub time_start: String,
    pub time_end: String,
    pub aggregate_kwh: f64,
    pub aggregate_kg_co2e: f64,
    pub issuer: String,
}

/// Detailed batch view with receipt IDs.
#[derive(Debug, Serialize)]
pub struct BatchDetail {
    #[serde(flatten)]
    pub summary: BatchSummary,
    pub receipt_ids: Vec<String>,
    pub commitment: Option<BatchCommitment>,
}

impl From<&ReceiptBatch> for BatchSummary {
    fn from(b: &ReceiptBatch) -> Self {
        BatchSummary {
            batch_id: b.batch_id.clone(),
            merkle_root: b.merkle_root.clone(),
            receipt_count: b.receipt_count,
            time_start: b.time_start.to_rfc3339(),
            time_end: b.time_end.to_rfc3339(),
            aggregate_kwh: b.aggregate_kwh,
            aggregate_kg_co2e: b.aggregate_kg_co2e,
            issuer: b.issuer.clone(),
        }
    }
}

/// Verify a receipt's inclusion in a committed batch.
async fn verify_receipt(
    State(state): State<LedgerVerifyState>,
    Path(receipt_id): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let store = state.store.read().await;

    let (receipt, batch_id, leaf_index) = store
        .receipts
        .get(&receipt_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    let tree = store
        .trees
        .get(batch_id)
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    let proof = tree
        .proof(*leaf_index)
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    let commitment = store
        .commitments
        .get(batch_id)
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    let batch = store
        .batches
        .get(batch_id)
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    let verified = proof.verify(&tree.root());

    Ok(Json(VerifyResponse {
        receipt: receipt.clone(),
        proof,
        batch: BatchSummary::from(batch),
        commitment: commitment.clone(),
        verified,
    }))
}

/// List committed batches.
async fn list_batches(State(state): State<LedgerVerifyState>) -> impl IntoResponse {
    let store = state.store.read().await;
    let summaries: Vec<BatchSummary> = store.batches.values().map(BatchSummary::from).collect();
    Json(summaries)
}

/// Get batch detail by ID.
async fn get_batch(
    State(state): State<LedgerVerifyState>,
    Path(batch_id): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let store = state.store.read().await;

    let batch = store.batches.get(&batch_id).ok_or(StatusCode::NOT_FOUND)?;

    let commitment = store.commitments.get(&batch_id).cloned();

    Ok(Json(BatchDetail {
        summary: BatchSummary::from(batch),
        receipt_ids: batch.receipt_ids.clone(),
        commitment,
    }))
}

// ============================================================================
// Receipt list endpoint
// ============================================================================

/// Query parameters for `GET /api/v1/ledger/receipts`.
#[derive(Debug, Deserialize)]
pub struct ListReceiptsParams {
    /// Maximum number of receipts to return (default 100, max 1000).
    pub limit: Option<usize>,
}

/// Summary of a single receipt (lightweight, no Merkle proof).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptSummary {
    pub receipt_id: String,
    pub qid: String,
    pub tenant_id: String,
    pub energy_joules_total: f64,
    pub kwh: f64,
    pub kg_co2e: f64,
    pub device_target: String,
    pub algorithm_type: String,
    pub timestamp_start: String,
    pub timestamp_end: String,
    pub batch_id: String,
}

/// List recent receipts.
async fn list_receipts(
    State(state): State<LedgerVerifyState>,
    Query(params): Query<ListReceiptsParams>,
) -> impl IntoResponse {
    let limit = params.limit.unwrap_or(100).min(1000);
    let store = state.store.read().await;

    let mut summaries: Vec<ReceiptSummary> = store
        .receipts
        .values()
        .map(|(receipt, batch_id, _)| ReceiptSummary {
            receipt_id: receipt.receipt_id.clone(),
            qid: receipt.qid.clone(),
            tenant_id: receipt.tenant_id.clone(),
            energy_joules_total: receipt.energy_joules_total,
            kwh: receipt.kwh,
            kg_co2e: receipt.kg_co2e,
            device_target: receipt.device_target.clone(),
            algorithm_type: receipt.algorithm_type.clone(),
            timestamp_start: receipt.timestamp_start.to_rfc3339(),
            timestamp_end: receipt.timestamp_end.to_rfc3339(),
            batch_id: batch_id.clone(),
        })
        .collect();

    // Sort by timestamp descending (most recent first).
    summaries.sort_by(|a, b| b.timestamp_start.cmp(&a.timestamp_start));
    summaries.truncate(limit);

    Json(summaries)
}

// ============================================================================
// Stats endpoint
// ============================================================================

/// Aggregate ledger statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerStats {
    pub total_receipts: usize,
    pub total_batches: usize,
    pub total_energy_joules: f64,
    pub total_kwh: f64,
    pub total_kg_co2e: f64,
    pub by_device: HashMap<String, f64>,
    pub by_algorithm: HashMap<String, f64>,
    pub by_stage: HashMap<String, f64>,
    pub oldest_receipt: Option<String>,
    pub newest_receipt: Option<String>,
}

/// Compute aggregate ledger statistics.
async fn ledger_stats(State(state): State<LedgerVerifyState>) -> impl IntoResponse {
    let store = state.store.read().await;

    let mut total_energy_joules = 0.0_f64;
    let mut total_kwh = 0.0_f64;
    let mut total_kg_co2e = 0.0_f64;
    let mut by_device: HashMap<String, f64> = HashMap::new();
    let mut by_algorithm: HashMap<String, f64> = HashMap::new();
    let mut by_stage: HashMap<String, f64> = HashMap::new();
    let mut oldest: Option<chrono::DateTime<chrono::Utc>> = None;
    let mut newest: Option<chrono::DateTime<chrono::Utc>> = None;

    for (receipt, _, _) in store.receipts.values() {
        total_energy_joules += receipt.energy_joules_total;
        total_kwh += receipt.kwh;
        total_kg_co2e += receipt.kg_co2e;

        *by_device
            .entry(receipt.device_target.clone())
            .or_insert(0.0) += receipt.energy_joules_total;
        *by_algorithm
            .entry(receipt.algorithm_type.clone())
            .or_insert(0.0) += receipt.energy_joules_total;

        for (stage_name, joules) in &receipt.energy_joules_by_stage {
            *by_stage.entry(stage_name.clone()).or_insert(0.0) += joules;
        }

        match oldest {
            None => oldest = Some(receipt.timestamp_start),
            Some(o) if receipt.timestamp_start < o => oldest = Some(receipt.timestamp_start),
            _ => {}
        }
        match newest {
            None => newest = Some(receipt.timestamp_end),
            Some(n) if receipt.timestamp_end > n => newest = Some(receipt.timestamp_end),
            _ => {}
        }
    }

    Json(LedgerStats {
        total_receipts: store.receipts.len(),
        total_batches: store.batches.len(),
        total_energy_joules,
        total_kwh,
        total_kg_co2e,
        by_device,
        by_algorithm,
        by_stage,
        oldest_receipt: oldest.map(|t| t.to_rfc3339()),
        newest_receipt: newest.map(|t| t.to_rfc3339()),
    })
}

/// Build the ledger verification router.
pub fn ledger_routes(state: LedgerVerifyState) -> Router {
    Router::new()
        .route(
            "/api/v1/ledger/receipts/{receipt_id}/verify",
            get(verify_receipt),
        )
        .route("/api/v1/ledger/receipts", get(list_receipts))
        .route("/api/v1/ledger/batches", get(list_batches))
        .route("/api/v1/ledger/batches/{batch_id}", get(get_batch))
        .route("/api/v1/ledger/stats", get(ledger_stats))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_state() -> LedgerVerifyState {
        LedgerVerifyState {
            store: Arc::new(RwLock::new(ReceiptStore::new())),
        }
    }

    fn make_receipt(id: &str, energy: f64, device: &str, algo: &str) -> LedgerEnergyReceipt {
        use chrono::{TimeZone, Utc};
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        LedgerEnergyReceipt {
            receipt_id: id.to_string(),
            qid: format!("q_{}", id),
            tenant_id: "default".to_string(),
            workload_tag: None,
            energy_joules_total: energy,
            energy_joules_by_stage: HashMap::new(),
            kwh: energy / 3_600_000.0,
            kg_co2e: energy / 3_600_000.0 * 0.4,
            grid_region: "US".to_string(),
            grid_factor_source: "test".to_string(),
            timestamp_start: ts,
            timestamp_end: ts + chrono::Duration::milliseconds(50),
            device_target: device.to_string(),
            algorithm_type: algo.to_string(),
        }
    }

    #[tokio::test]
    async fn test_list_receipts_empty() {
        let app = ledger_routes(test_state());
        let req = Request::builder()
            .uri("/api/v1/ledger/receipts")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1_000_000)
            .await
            .unwrap();
        let receipts: Vec<ReceiptSummary> = serde_json::from_slice(&body).unwrap();
        assert!(receipts.is_empty());
    }

    #[tokio::test]
    async fn test_list_receipts_with_data() {
        let state = test_state();
        {
            let mut store = state.store.write().await;
            let r = make_receipt("r1", 0.005, "cpu", "btree");
            store
                .receipts
                .insert("r1".to_string(), (r, "batch1".to_string(), 0));
        }
        let app = ledger_routes(state);
        let req = Request::builder()
            .uri("/api/v1/ledger/receipts?limit=10")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1_000_000)
            .await
            .unwrap();
        let receipts: Vec<ReceiptSummary> = serde_json::from_slice(&body).unwrap();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].receipt_id, "r1");
    }

    #[tokio::test]
    async fn test_stats_empty() {
        let app = ledger_routes(test_state());
        let req = Request::builder()
            .uri("/api/v1/ledger/stats")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1_000_000)
            .await
            .unwrap();
        let stats: LedgerStats = serde_json::from_slice(&body).unwrap();
        assert_eq!(stats.total_receipts, 0);
        assert_eq!(stats.total_batches, 0);
        assert!(stats.oldest_receipt.is_none());
    }

    #[tokio::test]
    async fn test_stats_with_data() {
        let state = test_state();
        {
            let mut store = state.store.write().await;
            let r1 = make_receipt("r1", 1000.0, "cpu", "btree");
            let r2 = make_receipt("r2", 2000.0, "gpu", "columnar");
            store
                .receipts
                .insert("r1".to_string(), (r1, "b1".to_string(), 0));
            store
                .receipts
                .insert("r2".to_string(), (r2, "b1".to_string(), 1));
        }
        let app = ledger_routes(state);
        let req = Request::builder()
            .uri("/api/v1/ledger/stats")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), 1_000_000)
            .await
            .unwrap();
        let stats: LedgerStats = serde_json::from_slice(&body).unwrap();
        assert_eq!(stats.total_receipts, 2);
        assert!((stats.total_energy_joules - 3000.0).abs() < 1e-10);
        assert_eq!(stats.by_device.len(), 2);
        assert!((stats.by_device["cpu"] - 1000.0).abs() < 1e-10);
        assert!((stats.by_device["gpu"] - 2000.0).abs() < 1e-10);
    }
}
