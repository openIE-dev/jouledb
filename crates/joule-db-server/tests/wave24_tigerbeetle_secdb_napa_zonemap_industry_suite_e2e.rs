//! Wave 24 End-to-End Verification:
//! Certifies `TigerBeetle` Linked Two-Phase Accounting + `VOPR` Deterministic Fault Simulator,
//! Goldman Sachs `SecDB` + Google `Napa` Reactive Risk/Freshness-SLA Engine,
//! `ClickBench` / `Vortex` `ZoneMap` + `SelectionVector` Late-Materialization Columnar Scanner,
//! and the Unified 7-Pillar 360° Industry Benchmark Suite (`POST /api/v1/bench/industry_suite`).

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use joule_db_core::tiger_ledger::{
    AccountFlags, LedgerAccount, LedgerTransfer, TigerDoubleEntryEngine, TransferError,
    TransferKind,
};
use joule_db_server::{Server, ServerConfig};
use serde_json::{Value, json};
use tower::ServiceExt;

async fn post_json(router: &axum::Router, uri: &str, payload: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .expect("build request");
    let resp = router.clone().oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let val: Value = serde_json::from_slice(&bytes).unwrap_or_else(|_| json!({}));
    (status, val)
}

#[tokio::test]
async fn test_wave24_tigerbeetle_vopr_secdb_napa_zonemap_and_seven_pillar_industry_suite() {
    // 1. Direct `TigerBeetle` Linked-Chain Rollback & Two-Phase Partial Settlement Proof
    let mut tb = TigerDoubleEntryEngine::new();
    tb.create_account(LedgerAccount::new(1, 840, 1, AccountFlags::default()))
        .expect("treasury");
    tb.create_account(LedgerAccount::new(
        2,
        840,
        10,
        AccountFlags {
            debits_must_not_exceed_credits: true,
            ..Default::default()
        },
    ))
    .expect("alice");
    tb.create_account(LedgerAccount::new(
        3,
        840,
        10,
        AccountFlags {
            debits_must_not_exceed_credits: true,
            ..Default::default()
        },
    ))
    .expect("bob");

    // Fund Alice with $1,000, then run a 2-leg `linked` transfer where Leg 1 ($400 Alice->Bob) is valid
    // but Leg 2 ($900 Alice->Bob) exceeds Alice's remaining $600 credits — BOTH legs must roll back!
    let r_fund = tb.submit_transfers(&[LedgerTransfer {
        id: 100,
        debit_account_id: 1,
        credit_account_id: 2,
        amount: 1_000,
        ledger: 840,
        code: 1,
        kind: TransferKind::DirectPost,
        linked: false,
    }]);
    assert_eq!(r_fund.committed, 1);

    let r_linked = tb.submit_transfers(&[
        LedgerTransfer {
            id: 101,
            debit_account_id: 2,
            credit_account_id: 3,
            amount: 400,
            ledger: 840,
            code: 2,
            kind: TransferKind::DirectPost,
            linked: true,
        },
        LedgerTransfer {
            id: 102,
            debit_account_id: 2,
            credit_account_id: 3,
            amount: 900,
            ledger: 840,
            code: 2,
            kind: TransferKind::DirectPost,
            linked: false,
        },
    ]);
    assert_eq!(r_linked.committed, 0);
    assert_eq!(r_linked.rejected, 2);
    assert!(matches!(
        r_linked.errors[0].1,
        TransferError::LinkedChainAborted { failed_id: 102 }
    ));
    assert_eq!(tb.account(2).expect("alice").debits_posted, 0);
    assert_eq!(tb.account(3).expect("bob").credits_posted, 0);
    assert!(r_linked.zero_sum_conserved);

    // Two-phase Pending Reserve ($500) -> Partial Post Settlement ($320): $180 released, $320 posted!
    let r_2pc = tb.submit_transfers(&[
        LedgerTransfer {
            id: 201,
            debit_account_id: 2,
            credit_account_id: 3,
            amount: 500,
            ledger: 840,
            code: 3,
            kind: TransferKind::PendingReserve,
            linked: false,
        },
        LedgerTransfer {
            id: 202,
            debit_account_id: 2,
            credit_account_id: 3,
            amount: 320,
            ledger: 840,
            code: 3,
            kind: TransferKind::PostPending { pending_id: 201 },
            linked: false,
        },
    ]);
    assert_eq!(r_2pc.committed, 2);
    assert_eq!(tb.account(2).expect("alice").debits_pending, 0);
    assert_eq!(tb.account(2).expect("alice").debits_posted, 320);
    assert!(r_2pc.zero_sum_conserved);

    // 2. Live HTTP Verification of All Wave 24 Routes + 7-Pillar 360° Industry Suite
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut cfg = ServerConfig::default();
    cfg.db_path = tmp.path().join("db").to_string_lossy().to_string();
    cfg.auth_enabled = false;
    cfg.rate_limiting_enabled = false;
    cfg.enable_tcp = false;
    cfg.enable_pgwire = false;
    cfg.enable_websocket = false;

    let srv = Server::new(cfg).expect("Server::new");
    let router = srv.router();

    let (st_vopr, vopr_v) = post_json(
        &router,
        "/api/v1/certify/tiger_vopr",
        json!({ "seeds": 48, "transfers_per_seed": 80 }),
    )
    .await;
    assert_eq!(st_vopr, StatusCode::OK, "tiger_vopr: {vopr_v}");
    assert_eq!(vopr_v["receipt"]["invariant_clean"].as_bool(), Some(true));
    assert_eq!(vopr_v["receipt"]["conservation_failures"].as_u64(), Some(0));

    let (st_zm, zm_v) = post_json(
        &router,
        "/api/v1/olap/zonemap_scan",
        json!({ "num_rows": 120_000, "range_lo": 60_000, "range_hi": 60_299 }),
    )
    .await;
    assert_eq!(st_zm, StatusCode::OK, "zonemap_scan: {zm_v}");
    assert_eq!(zm_v["receipt"]["matched_rows"].as_u64(), Some(300));
    assert!(
        zm_v["receipt"]["granule_skip_ratio"]
            .as_f64()
            .unwrap_or(0.0)
            > 0.97
    );

    let (st_sec, sec_v) = post_json(
        &router,
        "/api/v1/secdb/napa_eval",
        json!({ "num_desks": 80, "burst_ticks": 150, "max_staleness_ms": 50 }),
    )
    .await;
    assert_eq!(st_sec, StatusCode::OK, "secdb_napa: {sec_v}");
    assert_eq!(
        sec_v["single_tick_dirty_cone"]["nodes_recomputed"].as_u64(),
        Some(1)
    );
    assert_eq!(sec_v["napa_coalesced_writes"].as_u64(), Some(150));
    assert_eq!(sec_v["napa_lazy_refreshes"].as_u64(), Some(1));

    let (st_suite, suite_v) = post_json(
        &router,
        "/api/v1/bench/industry_suite",
        json!({
            "clickbench_rows": 100_000,
            "ann_vectors": 2_000,
            "tpcc_txns": 500,
        }),
    )
    .await;
    assert_eq!(st_suite, StatusCode::OK, "industry_suite: {suite_v}");
    assert_eq!(
        suite_v["all_seven_smoke_checks_passed"].as_bool(),
        Some(true)
    );
    assert_eq!(suite_v["energy_evidence"].as_str(), Some("est_j"));
    println!(
        "[Wave24 7-Pillar Industry Suite] {}",
        serde_json::to_string_pretty(&suite_v).unwrap_or_default()
    );
}
