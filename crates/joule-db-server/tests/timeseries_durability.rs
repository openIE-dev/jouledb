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
