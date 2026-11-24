//! Database Certification, Isolation Verification, Out-of-Core Morsel OLAP, `ZoneMap` Late
//! Materialization, `TigerBeetle` `VOPR` Ledger, `SecDB + Napa`, & 360° Industry Benchmark Routes.
//!
//! Surfaces JouleDB's Wave 23 & Wave 24 gap-closure and benchmark engines over HTTP:
//! - `POST /api/v1/certify/hermitage_elle` & `POST /api/v1/certify/elle_analyze`
//! - `POST /api/v1/certify/sqllogictest` & `POST /api/v1/certify/sqlancer`
//! - `POST /api/v1/certify/tiger_vopr`: `TigerBeetle` linked + two-phase pending + `VOPR` fault audit
//! - `POST /api/v1/olap/morsel_join` & `POST /api/v1/olap/morsel_groupby`
//! - `POST /api/v1/olap/zonemap_scan`: `ClickBench` / `Vortex` `ZoneMap` + `SelectionVector` scan
//! - `POST /api/v1/secdb/napa_eval`: Goldman `SecDB` dirty-cone DAG + Google `Napa` freshness SLA
//! - `POST /api/v1/bench/industry_suite`: Unified 7-pillar 360° industry benchmark runner

use crate::query::{QueryExecutor, QueryRequest};
use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use joule_db_core::concurrency::{
    ElleTxn, MvccTransactionManager, analyze_elle_history, run_hermitage_ssi_suite, to_elle_edn,
};
use joule_db_core::index::rabitq::{RaBitQConfig, RaBitQIndex};
use joule_db_core::tiger_ledger::run_vopr_deterministic_fault_campaign;
use joule_db_query::conformance_oracle::{
    run_sqlancer_tlp_and_norec_suite, run_sqllogictest_script,
};
use joule_db_query::dbsp::{IncrementalEquiJoin, IncrementalGroupSum, ZSet};
use joule_db_query::morsel_spill::{
    ColumnMorsel, MORSEL_ROWS, SpillAggSpec, chunk_into_morsels, execute_spilling_hash_aggregate,
    execute_spilling_hash_join,
};
use joule_db_query::secdb_napa::{NapaFreshnessController, SecDbFormula, SecDbReactiveGraph};
use joule_db_query::zonemap_scan::{ZoneMappedTable, ZonePredicate};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

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

#[derive(Debug, Deserialize)]
pub struct TigerVoprReq {
    #[serde(default = "default_vopr_seeds")]
    pub seeds: usize,
    #[serde(default = "default_vopr_transfers")]
    pub transfers_per_seed: usize,
}

#[derive(Debug, Deserialize)]
pub struct ZoneMapScanReq {
    #[serde(default = "default_zonemap_rows")]
    pub num_rows: usize,
    pub range_lo: i64,
    pub range_hi: i64,
}

#[derive(Debug, Deserialize)]
pub struct SecDbNapaReq {
    #[serde(default = "default_desks")]
    pub num_desks: usize,
    #[serde(default = "default_burst_ticks")]
    pub burst_ticks: usize,
    #[serde(default = "default_sla_ms")]
    pub max_staleness_ms: u64,
}

#[derive(Debug, Default, Deserialize)]
pub struct IndustrySuiteReq {
    #[serde(default = "default_zonemap_rows")]
    pub clickbench_rows: usize,
    #[serde(default = "default_ann_vectors")]
    pub ann_vectors: usize,
    #[serde(default = "default_tpcc_txns")]
    pub tpcc_txns: usize,
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

fn default_vopr_seeds() -> usize {
    64
}

fn default_vopr_transfers() -> usize {
    128
}

fn default_zonemap_rows() -> usize {
    100_000
}

fn default_desks() -> usize {
    64
}

fn default_burst_ticks() -> usize {
    200
}

fn default_sla_ms() -> u64 {
    50
}

fn default_ann_vectors() -> usize {
    4_000
}

fn default_tpcc_txns() -> usize {
    1_000
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

async fn handle_tiger_vopr(Json(req): Json<TigerVoprReq>) -> impl IntoResponse {
    let t0 = Instant::now();
    let receipt = run_vopr_deterministic_fault_campaign(req.seeds, req.transfers_per_seed);
    let elapsed_us = t0.elapsed().as_micros() as u64;
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "elapsed_us": elapsed_us,
            "receipt": receipt,
        })),
    )
}

async fn handle_zonemap_scan(Json(req): Json<ZoneMapScanReq>) -> impl IntoResponse {
    let n = req.num_rows.clamp(1_024, 2_000_000);
    let ts: Vec<Option<i64>> = (0..n).map(|i| Some(i as i64)).collect();
    let rev: Vec<f64> = (0..n).map(|i| ((i % 200) as f64) * 1.25).collect();
    let uid: Vec<u64> = (0..n).map(|i| (i % 500) as u64).collect();
    let tbl = match ZoneMappedTable::from_columns(&ts, &rev, &uid) {
        Ok(t) => t,
        Err(e) => return bad_req(e).into_response(),
    };
    let t0 = Instant::now();
    let receipt = tbl.scan_late_materialized(ZonePredicate::Between {
        lo: req.range_lo,
        hi: req.range_hi,
    });
    let elapsed_us = t0.elapsed().as_micros() as u64;
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "elapsed_us": elapsed_us,
            "receipt": receipt,
        })),
    )
        .into_response()
}

async fn handle_secdb_napa(Json(req): Json<SecDbNapaReq>) -> impl IntoResponse {
    let desks = req.num_desks.clamp(4, 1_024);
    let bursts = req.burst_ticks.clamp(1, 10_000);
    let mut dag = SecDbReactiveGraph::new();
    dag.upsert_source("fx_usd_jpy", 152.40, 0);
    for d in 0..desks {
        let s = format!("book_{d}_spot");
        dag.upsert_source(&s, 100.0 + (d as f64) * 0.5, 0);
        let _ = dag.upsert_derived(
            format!("book_{d}_call"),
            vec![s],
            SecDbFormula::OptionMarkApprox,
            0,
        );
    }
    let _ = dag.upsert_derived(
        "jpy_exposure",
        vec!["fx_usd_jpy".to_string()],
        SecDbFormula::LinearComb {
            weights: vec![250_000.0],
            bias: 0.0,
        },
        0,
    );

    let t0 = Instant::now();
    let _ = dag.stage_source_tick("fx_usd_jpy", 153.10, 5);
    let single_tick_rec = dag.propagate_dirty(5);

    let mut napa = NapaFreshnessController::new(req.max_staleness_ms, 0);
    let mut write_nj = 0u64;
    for step in 1..=bursts {
        write_nj += napa
            .on_write(
                &mut dag,
                "fx_usd_jpy",
                153.10 + (step as f64) * 0.01,
                5 + (step as u64),
            )
            .unwrap_or(0);
    }
    let (final_val, read_nj) = napa
        .query_with_sla(
            &mut dag,
            "jpy_exposure",
            5 + (bursts as u64) + req.max_staleness_ms + 10,
        )
        .unwrap_or((0.0, 0));
    let elapsed_us = t0.elapsed().as_micros() as u64;

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "elapsed_us": elapsed_us,
            "single_tick_dirty_cone": single_tick_rec,
            "napa_coalesced_writes": napa.coalesced_writes,
            "napa_lazy_refreshes": napa.lazy_refreshes,
            "final_exposure": final_val,
            "total_energy_nj": write_nj + read_nj,
        })),
    )
}

async fn handle_industry_suite(Json(req): Json<IndustrySuiteReq>) -> impl IntoResponse {
    let suite_t0 = Instant::now();

    // 1. `TPC-C` Micro-Mix (`NewOrder` + `Payment` Stock/Balance Conservation under Cahill `SSI`)
    let tpcc_n = req.tpcc_txns.clamp(50, 20_000);
    let tm = MvccTransactionManager::new();
    {
        let mut init = tm.begin_serializable();
        let _ = init.put(b"tpcc:wh_ytd", &0i64.to_le_bytes());
        for item in 0..16u32 {
            let k = format!("tpcc:stock:{item}");
            let _ = init.put(k.as_bytes(), &10_000i64.to_le_bytes());
        }
        let _ = init.commit();
    }
    let t_tpcc = Instant::now();
    let mut tpcc_committed = 0usize;
    let mut expected_ytd = 0i64;
    for i in 0..tpcc_n {
        let mut tx = tm.begin_serializable();
        let item_id = (i % 16) as u32;
        let sk = format!("tpcc:stock:{item_id}");
        let cur_stock = tx
            .get(sk.as_bytes())
            .ok()
            .flatten()
            .and_then(|b| b.try_into().ok().map(i64::from_le_bytes))
            .unwrap_or(0);
        let cur_ytd = tx
            .get(b"tpcc:wh_ytd")
            .ok()
            .flatten()
            .and_then(|b| b.try_into().ok().map(i64::from_le_bytes))
            .unwrap_or(0);
        let pay = ((i % 50) + 10) as i64;
        let ok = tx
            .put(sk.as_bytes(), &(cur_stock - 1).to_le_bytes())
            .is_ok()
            && tx
                .put(b"tpcc:wh_ytd", &(cur_ytd + pay).to_le_bytes())
                .is_ok()
            && tx.commit().is_ok();
        if ok {
            tpcc_committed += 1;
            expected_ytd += pay;
        }
    }
    let tpcc_us = t_tpcc.elapsed().as_micros().max(1) as u64;
    let final_ytd = tm
        .get(b"tpcc:wh_ytd")
        .and_then(|b| b.try_into().ok().map(i64::from_le_bytes))
        .unwrap_or(-1);

    // 2. `ClickBench` `ZoneMap` + `SelectionVector` Late Materialization (`100k` rows)
    let cb_rows = req.clickbench_rows.clamp(4_096, 500_000);
    let ts: Vec<Option<i64>> = (0..cb_rows).map(|i| Some(i as i64)).collect();
    let rev: Vec<f64> = (0..cb_rows).map(|i| ((i % 100) as f64) * 0.75).collect();
    let uid: Vec<u64> = (0..cb_rows).map(|i| (i % 400) as u64).collect();
    let z_tbl = match ZoneMappedTable::from_columns(&ts, &rev, &uid) {
        Ok(t) => t,
        Err(e) => return bad_req(e).into_response(),
    };
    let t_cb = Instant::now();
    let cb_rec = z_tbl.scan_late_materialized(ZonePredicate::Between {
        lo: (cb_rows / 2) as i64,
        hi: (cb_rows / 2 + 511) as i64,
    });
    let cb_us = t_cb.elapsed().as_micros().max(1) as u64;

    // 3. `TigerBeetle` Linked Two-Phase Transfers + `VOPR` Fault Simulation
    let t_tb = Instant::now();
    let vopr_rec = run_vopr_deterministic_fault_campaign(32, 100);
    let tb_us = t_tb.elapsed().as_micros().max(1) as u64;

    // 4. `SecDB + Napa` Reactive Risk DAG (`128` derivative nodes)
    let mut dag = SecDbReactiveGraph::new();
    dag.upsert_source("spot_nvda", 140.0, 0);
    for d in 0..64 {
        let s = format!("sec_{d}");
        dag.upsert_source(&s, 80.0 + d as f64, 0);
        let _ = dag.upsert_derived(
            format!("opt_{d}"),
            vec![s],
            SecDbFormula::OptionMarkApprox,
            0,
        );
    }
    let _ = dag.upsert_derived(
        "nvda_call",
        vec!["spot_nvda".to_string()],
        SecDbFormula::OptionMarkApprox,
        0,
    );
    let t_sec = Instant::now();
    let _ = dag.stage_source_tick("spot_nvda", 142.5, 1);
    let sec_rec = dag.propagate_dirty(1);
    let sec_us = t_sec.elapsed().as_micros().max(1) as u64;

    // 5. `Big-ANN` Selective Filtered `RaBitQ` 1-Bit `POPCNT` Vector Search (`128-D`)
    let n_vec = req.ann_vectors.clamp(256, 20_000);
    let dim = 128usize;
    let items: Vec<(u64, Vec<f32>)> = (0..n_vec as u64)
        .map(|id| {
            let v: Vec<f32> = (0..dim)
                .map(|j| (((id as usize * 13 + j * 7) as f32) * 0.031).sin())
                .collect();
            (id, v)
        })
        .collect();
    let target_id = 100u64; // Matches `id % 20 == 0` (`5%` selective filter!)
    let query_vec = items[target_id as usize].1.clone();
    let rabitq = RaBitQIndex::build_batch(RaBitQConfig::new(dim), &items);
    let t_ann = Instant::now();
    let hits = rabitq.search_filtered(&query_vec, 10, |id| id % 20 == 0);
    let ann_us = t_ann.elapsed().as_micros().max(1) as u64;
    let top10_hit = hits.iter().any(|h| h.id == target_id);

    // 6. `Nexmark` Continuous `DBSP` `ZSet` Auction Join + GroupSum ($O(|\Delta|)$)
    let mut join_op = IncrementalEquiJoin::<u64, String, i64>::new();
    let mut agg_op = IncrementalGroupSum::<String>::new();
    let auctions =
        ZSet::from_weighted((0..200u64).map(|aid| ((aid, format!("cat_{}", aid % 5)), 1i64)));
    let _ = join_op.step(&auctions, &ZSet::new());
    let t_nex = Instant::now();
    let bid_delta =
        ZSet::from_weighted((0..64u64).map(|b| (((b * 3) % 200, 100 + b as i64), 1i64)));
    let joined_delta = join_op.step(&ZSet::new(), &bid_delta);
    let cat_delta = joined_delta.map(|(_, cat, price)| (cat.clone(), *price));
    let changed_cats = agg_op.step(&cat_delta);
    let nex_us = t_nex.elapsed().as_micros().max(1) as u64;

    // 7. `Hermitage + Elle` Tarjan `SCC` Serializability Audit
    let herm_tm = MvccTransactionManager::new();
    let herm_rec = run_hermitage_ssi_suite(&herm_tm);

    let all_certified = (final_ytd == expected_ytd)
        && cb_rec.granule_skip_ratio > 0.90
        && vopr_rec.invariant_clean
        && sec_rec.compute_saved_ratio > 0.95
        && top10_hit
        && !changed_cats.is_empty()
        && herm_rec.elle_report.serializable_clean;

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "all_seven_pillars_certified": all_certified,
            "total_elapsed_ms": suite_t0.elapsed().as_secs_f64() * 1000.0,
            "tpcc_ssi": {
                "committed_txns": tpcc_committed,
                "ops_per_sec": (tpcc_committed as f64) * 1_000_000.0 / (tpcc_us as f64),
                "ytd_conserved": final_ytd == expected_ytd,
                "elapsed_us": tpcc_us,
            },
            "clickbench_zonemap": {
                "rows": cb_rows,
                "granule_skip_ratio": cb_rec.granule_skip_ratio,
                "matched_rows": cb_rec.matched_rows,
                "energy_nj": cb_rec.energy_nj,
                "elapsed_us": cb_us,
            },
            "tigerbeetle_vopr": {
                "transfers_simulated": vopr_rec.total_transfers_simulated,
                "invariant_clean": vopr_rec.invariant_clean,
                "torn_tails_detected": vopr_rec.torn_tails_detected,
                "bit_flips_detected": vopr_rec.bit_flips_detected,
                "elapsed_us": tb_us,
            },
            "secdb_napa_dag": {
                "total_nodes": sec_rec.total_nodes,
                "nodes_recomputed": sec_rec.nodes_recomputed,
                "compute_saved_ratio": sec_rec.compute_saved_ratio,
                "energy_nj": sec_rec.energy_nj,
                "elapsed_us": sec_us,
            },
            "big_ann_rabitq_filtered": {
                "vectors": n_vec,
                "compression_ratio_vs_fp32": rabitq.compression_ratio(),
                "selective_recall_hit": top10_hit,
                "elapsed_us": ann_us,
            },
            "nexmark_dbsp_stream": {
                "delta_bids": 64,
                "updated_categories": changed_cats.len(),
                "elapsed_us": nex_us,
            },
            "hermitage_elle_ssi": {
                "scenarios_passed": herm_rec.scenarios_passed,
                "elle_cycles": herm_rec.elle_report.scc_cycles_found,
                "serializable_clean": herm_rec.elle_report.serializable_clean,
            },
        })),
    )
        .into_response()
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
        .route("/api/v1/certify/tiger_vopr", post(handle_tiger_vopr))
        .route("/api/v1/olap/morsel_join", post(handle_morsel_join))
        .route("/api/v1/olap/morsel_groupby", post(handle_morsel_groupby))
        .route("/api/v1/olap/zonemap_scan", post(handle_zonemap_scan))
        .route("/api/v1/secdb/napa_eval", post(handle_secdb_napa))
        .route("/api/v1/bench/industry_suite", post(handle_industry_suite))
        .with_state(rt)
}
