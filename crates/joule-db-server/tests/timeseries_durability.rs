//! Time series survive a server restart.
//!
//! `TSWRITE` goes through the real `Server` query executor (the same one HTTP,
//! pgwire and MCP use) into `{db_path}/timeseries.wdb`. Dropping the server and
//! opening a new one on the same directory must return the identical series,
//! through SQL-style commands and through the MCP `jouledb://timeseries`
//! resources.

use joule_db_server::mcp_bridge::DatabaseToolHandler;
use joule_db_server::query::{QueryExecutor, QueryRequest, QueryResponse, TIMESERIES_FILE};
use joule_db_server::{Server, ServerConfig};
use serde_json::json;
use std::sync::Arc;

fn open_server(dir: &tempfile::TempDir) -> Server {
    let config = ServerConfig {
        http_addr: "127.0.0.1:0".to_string(),
        tcp_addr: "127.0.0.1:0".to_string(),
        db_path: dir.path().to_string_lossy().to_string(),
        enable_websocket: false,
        enable_tcp: false,
        enable_pgwire: false,
        enable_jwp: false,
        enable_webtransport: false,
        auth_enabled: false,
        ..ServerConfig::default()
    };
    Server::new(config).expect("server opens")
}

fn run(exec: &Arc<dyn QueryExecutor>, sql: &str) -> QueryResponse {
    let request = QueryRequest {
        sql: sql.to_string(),
        params: Default::default(),
        args: vec![],
        explain: false,
        limit: None,
        session_id: None,
        query_timeout_ms: None,
        branch_id: None,
        tenant_id: None,
    };
    exec.execute(&request)
        .unwrap_or_else(|e| panic!("{sql}: {e:?}"))
}

fn metrics(exec: &Arc<dyn QueryExecutor>) -> Vec<String> {
    let mut names: Vec<String> = run(exec, "TSLIST")
        .rows
        .iter()
        .map(|row| row[0].as_str().unwrap().to_string())
        .collect();
    names.sort();
    names
}

#[test]
fn timeseries_survive_server_restart() {
    let dir = tempfile::tempdir().unwrap();

    // ── First process lifetime: write, delete one series, read back. ──
    let before = {
        let server = open_server(&dir);
        let exec = server.query_executor();
        run(&exec, "TSWRITE cpu 0.5 1000 host=a");
        run(&exec, "TSWRITE cpu 0.75 2000 host=b");
        run(&exec, "TSWRITE cpu -3.25 3000");
        run(&exec, "TSWRITE mem 42 1500 host=a");
        run(&exec, "TSWRITE scratch 1 10");
        run(&exec, "TSDELETE scratch");
        assert_eq!(metrics(&exec), vec!["cpu", "mem"]);
        run(&exec, "TSQUERY cpu 0 10000").rows
    };
    assert_eq!(before.len(), 3);
    assert!(
        dir.path().join(TIMESERIES_FILE).is_file(),
        "series must live in {{db_path}}/{TIMESERIES_FILE}"
    );

    // ── Restart: a new server on the same directory sees identical data. ──
    {
        let server = open_server(&dir);
        let exec = server.query_executor();
        assert_eq!(metrics(&exec), vec!["cpu", "mem"], "deleted series stays deleted");

        let after = run(&exec, "TSQUERY cpu 0 10000").rows;
        assert_eq!(after, before, "points, values and tags unchanged across restart");
        assert_eq!(after[0], vec![json!(1000), json!(0.5), json!({"host": "a"})]);
        assert_eq!(after[2], vec![json!(3000), json!(-3.25), json!({})]);

        let mem = run(&exec, "TSQUERY mem 0 10000").rows;
        assert_eq!(mem, vec![vec![json!(1500), json!(42.0), json!({"host": "a"})]]);

        // Aggregates run on the in-memory store, which is rebuilt from disk.
        let sum = run(&exec, "TSAGGREGATE cpu 0 10000 10000 SUM").rows;
        let total: f64 = sum.iter().map(|r| r[1].as_f64().unwrap()).sum();
        assert!((total - (0.5 + 0.75 - 3.25)).abs() < 1e-12, "{sum:?}");

        // MCP resources go through the same executor and see durable data.
        let handler = DatabaseToolHandler::new(Arc::clone(&exec));
        let list = handler.read_resource("jouledb://timeseries");
        assert!(list.error.is_none(), "{:?}", list.error);
        let list = list.result.unwrap().to_string();
        assert!(list.contains("cpu") && list.contains("mem"), "{list}");
        assert!(!list.contains("scratch"), "{list}");

        let series = handler.read_resource("jouledb://timeseries/cpu?start=0&end=10000");
        assert!(series.error.is_none(), "{:?}", series.error);
        let series = series.result.unwrap().to_string();
        assert!(series.contains("0.75") && series.contains("-3.25"), "{series}");

        // Writes after a reopen append to the same durable series.
        run(&exec, "TSWRITE cpu 1.5 4000 host=c");
    }

    // ── Second restart: the post-reopen write is durable too. ──
    {
        let server = open_server(&dir);
        let exec = server.query_executor();
        let rows = run(&exec, "TSQUERY cpu 0 10000").rows;
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[..3], before[..]);
        assert_eq!(rows[3], vec![json!(4000), json!(1.5), json!({"host": "c"})]);
    }
}

#[test]
fn in_memory_executor_reports_non_durable_timeseries() {
    // Executors built without a database directory keep series in memory and
    // say so; opening a directory makes them durable.
    let exec = joule_db_server::query::SimpleQueryExecutor::new();
    assert!(!exec.timeseries_durable());

    let dir = tempfile::tempdir().unwrap();
    let exec = joule_db_server::query::SimpleQueryExecutor::open(dir.path());
    assert!(exec.timeseries_durable());
}

/// Writes to run against both a durable server and a memory-only executor.
/// `meta` is the name the first on-disk format used for its registry, and
/// `disk` / `disk::io` share a textual prefix; timestamps repeat on purpose.
const EDGE_WRITES: &[&str] = &[
    "TSWRITE meta 1 100 src=a",
    "TSWRITE meta 2 100 src=b",
    "TSWRITE meta 3 200",
    "TSWRITE cpu 0.5 100 host=a",
    "TSWRITE cpu 0.6 100 host=b",
    "TSWRITE cpu 0.7 100 host=c",
    "TSWRITE cpu 0.1 -50",
    "TSWRITE disk 9 300",
    "TSWRITE disk::io 7 300",
];

fn series(exec: &Arc<dyn QueryExecutor>, metric: &str) -> Vec<Vec<serde_json::Value>> {
    run(exec, &format!("TSQUERY {metric} -100000 100000")).rows
}

fn aggregate(exec: &Arc<dyn QueryExecutor>, metric: &str, agg: &str) -> Vec<Vec<serde_json::Value>> {
    run(exec, &format!("TSAGGREGATE {metric} -100000 100000 1000000 {agg}")).rows
}

#[test]
fn meta_series_and_duplicate_timestamps_survive_restart() {
    // Reference: the in-memory store, which keeps every point at a repeated
    // timestamp in write order.
    let memory: Arc<dyn QueryExecutor> = Arc::new(joule_db_server::query::SimpleQueryExecutor::new());
    for sql in EDGE_WRITES {
        run(&memory, sql);
    }
    let names = ["cpu", "disk", "disk::io", "meta"];
    let expected: Vec<_> = names.iter().map(|m| series(&memory, m)).collect();
    assert_eq!(expected[0].len(), 4, "memory keeps all cpu points: {:?}", expected[0]);
    assert_eq!(expected[3].len(), 3, "memory keeps all meta points: {:?}", expected[3]);
    assert_eq!(
        expected[0][1..].iter().map(|r| r[1].clone()).collect::<Vec<_>>(),
        vec![json!(0.5), json!(0.6), json!(0.7)],
        "same-timestamp points in write order"
    );

    let dir = tempfile::tempdir().unwrap();
    {
        let server = open_server(&dir);
        let exec = server.query_executor();
        for sql in EDGE_WRITES {
            run(&exec, sql);
        }
        assert_eq!(metrics(&exec), names);
        for (m, want) in names.iter().zip(&expected) {
            assert_eq!(&series(&exec, m), want, "{m} before restart");
        }
    }
    for round in 0..2 {
        let server = open_server(&dir);
        let exec = server.query_executor();
        // The memory-only executor is the reference for every series.
        let names = metrics(&memory);
        assert_eq!(metrics(&exec), names, "restart {round}");
        for m in &names {
            assert_eq!(series(&exec, m), series(&memory, m), "{m} after restart {round}");
            // The rebuilt in-memory store (aggregates) matches the reference.
            for agg in ["COUNT", "FIRST", "LAST", "SUM"] {
                assert_eq!(aggregate(&exec, m, agg), aggregate(&memory, m, agg), "{m} {agg}");
            }
        }
        if round == 0 {
            // Another point at an existing timestamp appends after the others,
            // and deleting `meta` touches no other series.
            for sql in ["TSWRITE cpu 0.8 100 host=d", "TSDELETE meta"] {
                run(&exec, sql);
                run(&memory, sql);
            }
        } else {
            assert_eq!(names, ["cpu", "disk", "disk::io"]);
            let cpu = series(&exec, "cpu");
            assert_eq!(cpu.len(), 5);
            assert_eq!(cpu.last().unwrap()[1], json!(0.8));
            assert!(series(&exec, "meta").is_empty());
        }
    }
}

#[test]
fn first_format_timeseries_file_still_loads() {
    // tests/fixtures/timeseries_v1.wdb.gz was written by the 7bccd15 code
    // (keys `__ts__::{metric}::{ts}` and `__ts__::meta::{metric}`), including
    // a series literally named `meta` and one named `disk::io`.
    use std::io::Read;
    let dir = tempfile::tempdir().unwrap();
    let gz = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/timeseries_v1.wdb.gz")).unwrap();
    let mut raw = Vec::new();
    flate2::read::GzDecoder::new(&gz[..]).read_to_end(&mut raw).unwrap();
    std::fs::write(dir.path().join(TIMESERIES_FILE), raw).unwrap();

    let check = |exec: &Arc<dyn QueryExecutor>| {
        assert_eq!(metrics(exec), ["cpu", "disk::io", "mem", "meta"]);
        assert_eq!(
            series(exec, "cpu"),
            vec![
                vec![json!(-5000), json!(-1.25), json!({})],
                vec![json!(1000), json!(10.5), json!({"host": "a"})],
                vec![json!(2000), json!(11.5), json!({})],
            ]
        );
        assert_eq!(series(exec, "mem"), vec![vec![json!(1000), json!(512.0), json!({"unit": "mb"})]]);
        assert_eq!(series(exec, "disk::io"), vec![vec![json!(3000), json!(7.0), json!({})]]);
        assert_eq!(series(exec, "meta"), vec![vec![json!(4000), json!(99.0), json!({})]]);
        assert_eq!(aggregate(exec, "cpu", "COUNT")[0][1], json!(3.0));
    };
    {
        let server = open_server(&dir);
        let exec = server.query_executor();
        check(&exec);
        run(&exec, "TSWRITE cpu 12.5 2000");
    }
    // Reopen: the migrated file is in the current format and still complete.
    let server = open_server(&dir);
    let exec = server.query_executor();
    assert_eq!(series(&exec, "cpu").len(), 4);
    assert_eq!(series(&exec, "cpu")[3], vec![json!(2000), json!(12.5), json!({})]);
}
