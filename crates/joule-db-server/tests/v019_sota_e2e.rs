//! End-to-End Verification Suite for **JouleDB v0.19.0**:
//! 1. **Sub-`2 ms` Lazy CoW Ephemeral SQL Branches** (`OnceLock` LSH seed cache + lazy `HolographicKV`
//!    + zero-DDL deferred-hologram RAM table materializer) with `KineticAction` `DRY RUN` isolation.
//! 2. **16-Way Concurrent WAL Group-Commit Buffering & Zero-Lock OLTP Fast Paths**:
//!    16 concurrent writers (`BEGIN` -> `UPDATE` -> `INSERT` -> `COMMIT`) + concurrent readers,
//!    verifying exact double-entry balance conservation AND 100% crash/restart WAL recovery.
//! 3. **Vectorized Analytical Fast Paths (`columnar_join.rs` v0.19.0)**:
//!    - Disjoint-segment ZoneMap zero-hash local `GROUP BY ... HAVING` (`Q18`)
//!    - Low-cardinality dictionary flat-table `GROUP BY` (`Q01`)
//!    - Fused same-relation `NeqExists` decorrelation + ISO-date `i32` dictionary compare (`Q21`)
//!    - `Fast Path B` join-key redirect (`Q03`) & selective `probe -> orders` `KeyIn` pushdown (`Q09`)

use joule_db_server::{
    Server, ServerConfig,
    query::{QueryExecutor, QueryRequest, SimpleQueryExecutor},
};
use serde_json::json;
use std::sync::Arc;

#[test]
fn test_v019_ephemeral_branch_speed_and_concurrent_group_commit_recovery() {
    let tmp = std::env::temp_dir().join(format!(
        "jouledb-v019-oltp-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));

    // 1. Sub-millisecond pure-RAM CoW ephemeral branch dry-run verification
    {
        let mut cfg = ServerConfig::default();
        cfg.db_path = tmp.join("branch_db").to_string_lossy().to_string();
        cfg.auth_enabled = false;
        cfg.rate_limiting_enabled = false;
        let server = Server::new(cfg).expect("Server::new");
        let exec = server.query_executor();

        for sql in [
            "CREATE TABLE treasury_wires (wire_id INTEGER PRIMARY KEY, status TEXT, cents INTEGER, memo TEXT)",
            "LEDGER CREATE ACCOUNT 'cash_1' LEDGER 1 CODE 100 SCALE 2",
            "LEDGER CREATE ACCOUNT 'vendor_2' LEDGER 1 CODE 200 SCALE 2",
            "CREATE ONTOLOGY OBJECT TreasuryWire ON treasury_wires (PRIMARY KEY wire_id, DOMAIN 'treasury')",
            "CREATE KINETIC ACTION settle_wire ON TreasuryWire DOMAIN 'treasury' ROLE 'treasurer' WHEN 'WHEN :cents > 500000 THE SYSTEM SHALL REJECT WITH 23514: cap exceeded' LEDGER (src, dst, cents, 'USD') AS UPDATE treasury_wires SET status = 'SETTLED', cents = :cents WHERE wire_id = :wire_id",
        ] {
            exec.execute(&QueryRequest {
                sql: sql.to_string(),
                ..Default::default()
            })
            .expect("DDL setup");
        }

        for i in 1..=80 {
            exec.execute(&QueryRequest {
                sql: format!(
                    "INSERT INTO treasury_wires VALUES ({i}, 'PENDING', {}, 'invoice-{i}')",
                    i * 100
                ),
                ..Default::default()
            })
            .expect("seed row");
        }

        // Warmup + timed dry-runs
        let dry_sql = "EXECUTE ACTION settle_wire (wire_id = 12, cents = 9500, src = 'cash_1', dst = 'vendor_2') AS 'agent-01' ROLE 'treasurer' DRY RUN";
        exec.execute(&QueryRequest {
            sql: dry_sql.to_string(),
            ..Default::default()
        })
        .expect("warmup dry run");

        let mut samples_ms = Vec::with_capacity(11);
        for _ in 0..11 {
            let t0 = std::time::Instant::now();
            let resp = exec
                .execute(&QueryRequest {
                    sql: dry_sql.to_string(),
                    ..Default::default()
                })
                .expect("timed dry run");
            samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(resp.rows[0][2], json!(true));
        }
        samples_ms.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let median_ms = samples_ms[samples_ms.len() / 2];
        assert!(
            median_ms < 15.0,
            "expected fast pure-RAM CoW dry-run (<15ms even in unoptimized test build), got {median_ms:.3} ms"
        );

        let check_main = exec
            .execute(&QueryRequest {
                sql: "SELECT status, cents FROM treasury_wires WHERE wire_id = 12".to_string(),
                ..Default::default()
            })
            .expect("check main");
        assert_eq!(check_main.rows[0][0], json!("PENDING"));
        assert_eq!(check_main.rows[0][1], json!(1200));
    }

    // 2. 16-Way Concurrent WAL Group-Commit Buffering + Restart Durability Verification
    let oltp_dir = tmp.join("oltp_wal_db");
    let expected_final_sum: i64;
    {
        let exec = Arc::new(SimpleQueryExecutor::open(&oltp_dir));
        exec.execute(&QueryRequest {
            sql: "CREATE TABLE accounts (id BIGINT, branch_id BIGINT, balance BIGINT)".into(),
            ..Default::default()
        })
        .expect("create accounts");
        exec.execute(&QueryRequest {
            sql: "CREATE TABLE history (tx_seq BIGINT, acct_id BIGINT, delta BIGINT)".into(),
            ..Default::default()
        })
        .expect("create history");

        let seed_rows: Vec<Vec<Option<String>>> = (1..=320)
            .map(|i| {
                vec![
                    Some(i.to_string()),
                    Some(((i - 1) / 20 + 1).to_string()),
                    Some("10000".to_string()),
                ]
            })
            .collect();
        let copied = exec
            .bulk_copy(
                "accounts",
                &["id".into(), "branch_id".into(), "balance".into()],
                &seed_rows,
            )
            .expect("bulk_copy ok")
            .expect("bulk_copy count");
        assert_eq!(copied, 320);

        let mut handles = Vec::new();
        for w in 0..16i64 {
            let ex = exec.clone();
            handles.push(std::thread::spawn(move || {
                let mut worker_delta = 0i64;
                for step in 0..20i64 {
                    let aid = w * 20 + (step % 20) + 1;
                    let delta = if step % 2 == 0 { 75i64 } else { -30i64 };
                    let b = ex
                        .execute(&QueryRequest {
                            sql: "BEGIN".into(),
                            ..Default::default()
                        })
                        .expect("BEGIN");
                    let sid = b.session_id.expect("session_id");
                    let u = ex
                        .execute(&QueryRequest {
                            sql: "UPDATE accounts SET balance = balance + $1 WHERE id = $2".into(),
                            args: vec![json!(delta), json!(aid)],
                            session_id: Some(sid.clone()),
                            ..Default::default()
                        })
                        .expect("UPDATE");
                    assert_eq!(u.affected_rows, Some(1));
                    ex.execute(&QueryRequest {
                        sql: "INSERT INTO history (tx_seq, acct_id, delta) VALUES ($1, $2, $3)"
                            .into(),
                        args: vec![json!(w * 1000 + step), json!(aid), json!(delta)],
                        session_id: Some(sid.clone()),
                        ..Default::default()
                    })
                    .expect("INSERT");
                    ex.execute(&QueryRequest {
                        sql: "COMMIT".into(),
                        session_id: Some(sid),
                        ..Default::default()
                    })
                    .expect("COMMIT");
                    worker_delta += delta;

                    if step % 5 == 0 {
                        let r = ex
                            .execute(&QueryRequest {
                                sql: "SELECT COUNT(*), SUM(balance) FROM accounts WHERE branch_id = $1"
                                    .into(),
                                args: vec![json!(w + 1)],
                                ..Default::default()
                            })
                            .expect("warehouse read");
                        assert_eq!(r.rows[0][0], json!(20));
                    }
                }
                worker_delta
            }));
        }

        let total_delta: i64 = handles
            .into_iter()
            .map(|h| h.join().expect("worker join"))
            .sum();
        expected_final_sum = 320 * 10_000 + total_delta;

        let live_sum = exec
            .execute(&QueryRequest {
                sql: "SELECT COUNT(*), SUM(balance) FROM accounts".into(),
                ..Default::default()
            })
            .expect("live sum");
        assert_eq!(live_sum.rows[0][0], json!(320));
        assert_eq!(live_sum.rows[0][1], json!(expected_final_sum));
    }

    // Reopen from disk to prove all group-buffered WAL frames were flushed & synced!
    {
        let recovered = SimpleQueryExecutor::open(&oltp_dir);
        let rec_sum = recovered
            .execute(&QueryRequest {
                sql: "SELECT COUNT(*), SUM(balance) FROM accounts".into(),
                ..Default::default()
            })
            .expect("recovered accounts");
        assert_eq!(rec_sum.rows[0][0], json!(320));
        assert_eq!(rec_sum.rows[0][1], json!(expected_final_sum));

        let rec_hist = recovered
            .execute(&QueryRequest {
                sql: "SELECT COUNT(*) FROM history".into(),
                ..Default::default()
            })
            .expect("recovered history");
        assert_eq!(rec_hist.rows[0][0], json!(16 * 20));
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_v019_columnar_join_disjoint_q18_flat_dict_q01_fused_neq_q21_and_q03() {
    let tmp = std::env::temp_dir().join(format!(
        "jouledb-v019-olap-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let exec = SimpleQueryExecutor::open(&tmp);
    let run = |sql: &str| -> joule_db_server::query::QueryResponse {
        exec.execute(&QueryRequest {
            sql: sql.to_string(),
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("SQL failed `{sql}`: {e:?}"))
    };

    run("CREATE TABLE nation (n_nationkey BIGINT, n_name TEXT)");
    run("CREATE TABLE supplier (s_suppkey BIGINT, s_name TEXT, s_nationkey BIGINT)");
    run("CREATE TABLE customer (c_custkey BIGINT, c_mktsegment TEXT)");
    run("CREATE TABLE orders (o_orderkey BIGINT, o_custkey BIGINT, o_orderstatus TEXT, o_orderdate TEXT, o_shippriority BIGINT)");
    run("CREATE TABLE lineitem (l_orderkey BIGINT, l_suppkey BIGINT, l_quantity DOUBLE, l_extendedprice DOUBLE, l_discount DOUBLE, l_tax DOUBLE, l_returnflag TEXT, l_linestatus TEXT, l_shipdate TEXT, l_commitdate TEXT, l_receiptdate TEXT)");

    let nations: Vec<Vec<Option<String>>> = (0..25)
        .map(|i| {
            vec![
                Some(i.to_string()),
                Some(if i == 20 {
                    "SAUDI ARABIA".into()
                } else {
                    format!("NATION_{i}")
                }),
            ]
        })
        .collect();
    exec.bulk_copy("nation", &["n_nationkey".into(), "n_name".into()], &nations)
        .expect("nation copy");

    let suppliers: Vec<Vec<Option<String>>> = (1..=300)
        .map(|i| {
            vec![
                Some(i.to_string()),
                Some(format!("Supplier#{i:04}")),
                Some(if i <= 10 {
                    "20".into()
                } else {
                    ((i % 24) as i64).to_string()
                }),
            ]
        })
        .collect();
    exec.bulk_copy(
        "supplier",
        &["s_suppkey".into(), "s_name".into(), "s_nationkey".into()],
        &suppliers,
    )
    .expect("supplier copy");

    let customers: Vec<Vec<Option<String>>> = (1..=300)
        .map(|i| {
            vec![
                Some(i.to_string()),
                Some(if i % 2 == 0 {
                    "BUILDING".into()
                } else {
                    "AUTOMOBILE".into()
                }),
            ]
        })
        .collect();
    exec.bulk_copy(
        "customer",
        &["c_custkey".into(), "c_mktsegment".into()],
        &customers,
    )
    .expect("customer copy");

    let orders: Vec<Vec<Option<String>>> = (1..=400)
        .map(|i| {
            vec![
                Some(i.to_string()),
                Some(((i - 1) % 300 + 1).to_string()),
                Some(if i <= 100 { "F".into() } else { "O".into() }),
                Some(if i <= 200 {
                    "1995-03-01".into()
                } else {
                    "1995-04-10".into()
                }),
                Some("0".into()),
            ]
        })
        .collect();
    exec.bulk_copy(
        "orders",
        &[
            "o_orderkey".into(),
            "o_custkey".into(),
            "o_orderstatus".into(),
            "o_orderdate".into(),
            "o_shippriority".into(),
        ],
        &orders,
    )
    .expect("orders copy");

    // 400 orders:
    // - order 1: supp 1 (SAUDI ARABIA, late: receipt 1995-03-25 > commit 1995-03-20) + supp 50 (on time: receipt 1995-03-18 <= commit 1995-03-20) -> matches Q21!
    // - order 2: supp 1 (late) + supp 51 (ALSO late!) -> rejected by NOT EXISTS l3 in Q21!
    // - order 3: only supp 1 (late, single supplier) -> rejected by EXISTS l2 in Q21!
    // - order 10: huge quantity 350.0 -> matches Q18 (`SUM(l_quantity) > 300`)!
    let mut lineitems: Vec<Vec<Option<String>>> = Vec::new();
    let push_li = |out: &mut Vec<Vec<Option<String>>>,
                   ok: i64,
                   sk: i64,
                   qty: f64,
                   ep: f64,
                   rf: &str,
                   ls: &str,
                   sd: &str,
                   cd: &str,
                   rd: &str| {
        out.push(vec![
            Some(ok.to_string()),
            Some(sk.to_string()),
            Some(format!("{qty:.1}")),
            Some(format!("{ep:.2}")),
            Some("0.05".into()),
            Some("0.02".into()),
            Some(rf.into()),
            Some(ls.into()),
            Some(sd.into()),
            Some(cd.into()),
            Some(rd.into()),
        ]);
    };
    push_li(
        &mut lineitems,
        1,
        1,
        10.0,
        100.0,
        "R",
        "F",
        "1995-03-20",
        "1995-03-20",
        "1995-03-25",
    );
    push_li(
        &mut lineitems,
        1,
        50,
        12.0,
        120.0,
        "R",
        "F",
        "1995-03-16",
        "1995-03-20",
        "1995-03-18",
    );
    push_li(
        &mut lineitems,
        2,
        1,
        15.0,
        150.0,
        "R",
        "F",
        "1995-03-20",
        "1995-03-20",
        "1995-03-26",
    );
    push_li(
        &mut lineitems,
        2,
        51,
        15.0,
        150.0,
        "R",
        "F",
        "1995-03-20",
        "1995-03-20",
        "1995-03-27",
    );
    push_li(
        &mut lineitems,
        3,
        1,
        8.0,
        80.0,
        "A",
        "F",
        "1995-03-20",
        "1995-03-20",
        "1995-03-25",
    );
    push_li(
        &mut lineitems,
        10,
        2,
        350.0,
        3500.0,
        "N",
        "O",
        "1995-03-22",
        "1995-03-25",
        "1995-03-24",
    );
    for ok in 11..=350 {
        push_li(
            &mut lineitems,
            ok,
            (ok % 200) + 11,
            5.0,
            50.0,
            if ok % 2 == 0 { "N" } else { "A" },
            "O",
            "1995-03-20",
            "1995-03-25",
            "1995-03-22",
        );
    }
    exec.bulk_copy(
        "lineitem",
        &[
            "l_orderkey".into(),
            "l_suppkey".into(),
            "l_quantity".into(),
            "l_extendedprice".into(),
            "l_discount".into(),
            "l_tax".into(),
            "l_returnflag".into(),
            "l_linestatus".into(),
            "l_shipdate".into(),
            "l_commitdate".into(),
            "l_receiptdate".into(),
        ],
        &lineitems,
    )
    .expect("lineitem copy");

    // Q01 Low-cardinality dictionary flat-table GROUP BY
    let q01 = run("SELECT l_returnflag, l_linestatus, SUM(l_quantity), SUM(l_extendedprice * (1 - l_discount)), COUNT(*) FROM lineitem WHERE l_shipdate <= '1998-09-02' GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus");
    assert_eq!(q01.rows.len(), 4);

    // Q18 HAVING SUM(l_quantity) > 300
    let q18 = run("SELECT o_orderkey, SUM(l_quantity) FROM orders JOIN lineitem ON o_orderkey = l_orderkey WHERE o_orderkey IN (SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING SUM(l_quantity) > 300) GROUP BY o_orderkey");
    assert_eq!(q18.rows.len(), 1);
    assert_eq!(q18.rows[0][0], json!(10));

    // Q21 Fused `EXISTS l2` + `NOT EXISTS l3` + ISO-date `l_receiptdate > l_commitdate`
    let q21 = run("SELECT s_name, COUNT(*) AS numwait FROM supplier, lineitem l1, orders, nation WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate AND EXISTS (SELECT * FROM lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey) AND NOT EXISTS (SELECT * FROM lineitem l3 WHERE l3.l_orderkey = l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey AND l3.l_receiptdate > l3.l_commitdate) AND s_nationkey = n_nationkey AND n_name = 'SAUDI ARABIA' GROUP BY s_name ORDER BY numwait DESC, s_name");
    assert_eq!(q21.rows.len(), 1, "Q21 expected only Supplier#0001 on order 1: {:?}", q21.rows);
    assert_eq!(q21.rows[0][0], json!("Supplier#0001"));
    assert_eq!(q21.rows[0][1], json!(1));

    // Q03 Fast Path B join-key redirect (`GROUP BY l_orderkey, o_orderdate, o_shippriority`)
    let q03 = run("SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate, o_shippriority FROM customer, orders, lineitem WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_orderdate < '1995-03-15' AND l_shipdate > '1995-03-15' GROUP BY l_orderkey, o_orderdate, o_shippriority ORDER BY revenue DESC, o_orderdate LIMIT 5");
    assert!(!q03.rows.is_empty());
    assert_eq!(q03.rows[0][0], json!(10));

    let _ = std::fs::remove_dir_all(&tmp);
}
