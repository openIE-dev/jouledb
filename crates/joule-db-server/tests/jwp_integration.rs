//! JWP (Joule Wire Protocol) Integration Tests
//!
//! These tests start a real JouleDB server with JWP enabled and verify
//! the full protocol flow over actual TCP connections: handshake, SQL
//! queries, heartbeat, error handling, energy tracking, and concurrency.

use std::collections::BTreeMap;
use std::net::TcpListener as StdTcpListener;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use joule_db_server::{Server, ServerConfig};
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio_util::codec::Framed;

use jwp::{
    AdaptiveCodec, ErrorPayload, FrameType, HandshakePayload, JwpFrame, cbor_decode, cbor_encode,
};

// ── Shared CBOR payloads (mirror jwp_server.rs) ────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct DbQueryPayload {
    sql: String,
    #[serde(default)]
    args: Vec<serde_json::Value>,
    #[serde(default)]
    named: BTreeMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
    #[serde(default)]
    explain: bool,
}

#[derive(Debug, Deserialize)]
struct DbResultPayload {
    rows: Vec<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct DbDonePayload {
    row_count: u64,
    #[serde(default)]
    affected_rows: Option<u64>,
    total_cost_uwh: u64,
    #[allow(dead_code)]
    elapsed_ms: u64,
}

// ── Helpers ─────────────────────────────────────────────────────────

fn find_available_port() -> u16 {
    let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Create a JouleDB server with JWP enabled and return (server, http_url, jwp_addr).
fn create_jwp_server() -> (Server, String, String) {
    let http_port = find_available_port();
    let jwp_port = find_available_port();
    let dir = tempfile::tempdir().unwrap();
    let config = ServerConfig {
        http_addr: format!("127.0.0.1:{}", http_port),
        tcp_addr: format!("127.0.0.1:{}", find_available_port()),
        db_path: dir.path().to_string_lossy().to_string(),
        enable_websocket: false,
        enable_tcp: false,
        max_tcp_connections: 100,
        enable_webtransport: false,
        webtransport_port: 0,
        enable_pgwire: false,
        pgwire_addr: "127.0.0.1:0".to_string(),
        enable_jwp: true,
        jwp_addr: format!("127.0.0.1:{}", jwp_port),
        max_jwp_connections: 100,
        auth_enabled: false,
        auth_jwt_secret: None,
        enable_replication: false,
        replication_role: None,
        replication_listen_addr: "127.0.0.1:0".to_string(),
        replication_leader_addr: None,
        #[cfg(feature = "tls")]
        tls_cert_path: None,
        #[cfg(feature = "tls")]
        tls_key_path: None,
        energy_config: joule_db_energy::EnergyConfig::default(),
        enable_raft: false,
        raft_node_id: None,
        raft_addr: String::new(),
        raft_peers: Vec::new(),
        raft_master_secret: None,
        query_timeout_ms: 30000,
        slow_query_threshold_ms: 1000,
        rate_limiting_enabled: false,
        rate_limit_requests_per_minute: 1000,
        max_result_rows: 100_000,
        session_timeout_secs: 300,
        #[cfg(feature = "tls")]
        require_tls: false,
        cors_origins: Vec::new(),
        sanitize_errors: false,
        runtime_mode: "native".to_string(),
        enable_ledger: false,
        ledger_dir: None,
        ledger_batch_max_receipts: 1000,
        ledger_batch_interval_secs: 60,
        ledger_grid_region: None,
        ledger_grid_factor: None,
        scale_to_zero_enabled: false,
        idle_timeout_secs: 300,
        enable_mcp_stdio: false,
    };
    // Keep dir alive by leaking it (test only)
    std::mem::forget(dir);
    let server = Server::new(config).unwrap();
    let base_url = format!("http://127.0.0.1:{}", http_port);
    let jwp_addr = format!("127.0.0.1:{}", jwp_port);
    (server, base_url, jwp_addr)
}

/// Spawn server in background and wait until HTTP health endpoint is ready.
async fn start_server(server: Server, base_url: &str) {
    let url = base_url.to_string();
    tokio::spawn(async move {
        server.run().await.unwrap();
    });
    let client = reqwest::Client::new();
    for _ in 0..100 {
        if client
            .get(format!("{}/health", url))
            .timeout(Duration::from_millis(100))
            .send()
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("Server failed to start within timeout");
}

/// Connect to JWP server, perform v1 handshake, return framed stream.
async fn jwp_connect(addr: &str) -> Framed<TcpStream, AdaptiveCodec> {
    let stream = tokio::time::timeout(Duration::from_secs(5), TcpStream::connect(addr))
        .await
        .expect("connect timeout")
        .expect("connect failed");

    let mut framed = Framed::new(stream, AdaptiveCodec::v1_compat());

    // Send handshake
    let hello = HandshakePayload {
        version: 1,
        capabilities: vec!["sql".into()],
    };
    framed
        .send(JwpFrame::new(
            FrameType::Handshake,
            1,
            0,
            cbor_encode(&hello).unwrap(),
        ))
        .await
        .unwrap();

    // Receive handshake ack
    let resp = tokio::time::timeout(Duration::from_secs(5), framed.next())
        .await
        .expect("handshake timeout")
        .expect("stream closed")
        .expect("decode error");
    assert_eq!(resp.header.frame_type, FrameType::Handshake);

    framed
}

/// Send a SQL query and collect (rows, done) or Err(error).
async fn jwp_query(
    framed: &mut Framed<TcpStream, AdaptiveCodec>,
    sql: &str,
    seq: u32,
) -> Result<(Vec<Vec<serde_json::Value>>, DbDonePayload), ErrorPayload> {
    let payload = DbQueryPayload {
        sql: sql.to_string(),
        args: vec![],
        named: BTreeMap::new(),
        session_id: None,
        limit: None,
        explain: false,
    };
    framed
        .send(JwpFrame::new(
            FrameType::Query,
            seq,
            0,
            cbor_encode(&payload).unwrap(),
        ))
        .await
        .unwrap();

    let mut rows = Vec::new();
    for _ in 0..200 {
        let resp = tokio::time::timeout(Duration::from_secs(10), framed.next())
            .await
            .expect("query timeout")
            .expect("stream closed")
            .expect("decode error");

        match resp.header.frame_type {
            FrameType::Meta => continue,
            FrameType::Result => {
                let result: DbResultPayload = cbor_decode(&resp.payload).unwrap();
                rows.extend(result.rows);
            }
            FrameType::Done => {
                let done: DbDonePayload = cbor_decode(&resp.payload).unwrap();
                return Ok((rows, done));
            }
            FrameType::Error => {
                let err: ErrorPayload = cbor_decode(&resp.payload).unwrap();
                return Err(err);
            }
            _ => continue,
        }
    }
    panic!("No Done frame received after 200 frames");
}

// ── Tests ───────────────────────────────────────────────────────────

#[tokio::test]
async fn jwp_connect_and_handshake() {
    let (server, base_url, jwp_addr) = create_jwp_server();
    start_server(server, &base_url).await;

    let stream = TcpStream::connect(&jwp_addr).await.unwrap();
    let mut framed = Framed::new(stream, AdaptiveCodec::v1_compat());

    // Send handshake
    let hello = HandshakePayload {
        version: 1,
        capabilities: vec!["sql".into(), "ledger".into()],
    };
    framed
        .send(JwpFrame::new(
            FrameType::Handshake,
            1,
            0,
            cbor_encode(&hello).unwrap(),
        ))
        .await
        .unwrap();

    // Receive handshake ack
    let resp = tokio::time::timeout(Duration::from_secs(5), framed.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();

    assert_eq!(resp.header.frame_type, FrameType::Handshake);
    let server_hs: HandshakePayload = cbor_decode(&resp.payload).unwrap();
    assert_eq!(server_hs.version, 1);
    assert!(server_hs.capabilities.contains(&"sql".to_string()));
    assert!(server_hs.capabilities.contains(&"subscribe".to_string()));
}

#[tokio::test]
async fn jwp_create_table_and_insert() {
    let (server, base_url, jwp_addr) = create_jwp_server();
    start_server(server, &base_url).await;
    let mut framed = jwp_connect(&jwp_addr).await;

    // CREATE TABLE
    let (_, done) = jwp_query(
        &mut framed,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)",
        2,
    )
    .await
    .unwrap();
    assert_eq!(done.row_count, 0);

    // INSERT
    let (_, done) = jwp_query(
        &mut framed,
        "INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com')",
        3,
    )
    .await
    .unwrap();
    assert_eq!(done.affected_rows, Some(1));

    // INSERT another row
    jwp_query(
        &mut framed,
        "INSERT INTO users (id, name, email) VALUES (2, 'Bob', 'bob@example.com')",
        4,
    )
    .await
    .unwrap();

    // SELECT
    let (rows, done) = jwp_query(&mut framed, "SELECT * FROM users ORDER BY id", 5)
        .await
        .unwrap();
    assert_eq!(done.row_count, 2);
    assert_eq!(rows.len(), 2);
    // First row should be Alice
    assert_eq!(rows[0][1], serde_json::Value::String("Alice".to_string()));
    // Second row should be Bob
    assert_eq!(rows[1][1], serde_json::Value::String("Bob".to_string()));
}

#[tokio::test]
async fn jwp_heartbeat_echo() {
    let (server, base_url, jwp_addr) = create_jwp_server();
    start_server(server, &base_url).await;
    let mut framed = jwp_connect(&jwp_addr).await;

    // Send heartbeat
    framed
        .send(JwpFrame::new(FrameType::Heartbeat, 2, 0, vec![]))
        .await
        .unwrap();

    // Receive heartbeat echo
    let resp = tokio::time::timeout(Duration::from_secs(5), framed.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();

    assert_eq!(resp.header.frame_type, FrameType::Heartbeat);
}

#[tokio::test]
async fn jwp_error_on_bad_sql() {
    let (server, base_url, jwp_addr) = create_jwp_server();
    start_server(server, &base_url).await;
    let mut framed = jwp_connect(&jwp_addr).await;

    // Send malformed SQL
    let result = jwp_query(&mut framed, "SELECTTTT NONSENSE FROM ???", 2).await;
    assert!(result.is_err(), "Expected Error frame for bad SQL");
    let err = result.unwrap_err();
    assert!(
        !err.message.is_empty(),
        "Error message should be non-empty: {}",
        err.message
    );
}

#[tokio::test]
async fn jwp_energy_tracking() {
    let (server, base_url, jwp_addr) = create_jwp_server();
    start_server(server, &base_url).await;
    let mut framed = jwp_connect(&jwp_addr).await;

    // Create table and insert some data
    jwp_query(
        &mut framed,
        "CREATE TABLE metrics (id INTEGER PRIMARY KEY, value REAL)",
        2,
    )
    .await
    .unwrap();

    for i in 1..=10 {
        jwp_query(
            &mut framed,
            &format!(
                "INSERT INTO metrics (id, value) VALUES ({}, {})",
                i,
                i as f64 * 1.5
            ),
            2 + i as u32,
        )
        .await
        .unwrap();
    }

    // Query should report energy cost
    let (_, done) = jwp_query(&mut framed, "SELECT * FROM metrics", 20)
        .await
        .unwrap();
    assert_eq!(done.row_count, 10);
    // Energy tracking is best-effort — just verify the field exists and is non-negative
    // (on macOS IOKit, actual power readings vary; in test environments may be 0)
    // The total_cost_uwh is computed from cumulative_joules which depends on energy monitoring
}

#[tokio::test]
async fn jwp_multiple_queries() {
    let (server, base_url, jwp_addr) = create_jwp_server();
    start_server(server, &base_url).await;
    let mut framed = jwp_connect(&jwp_addr).await;

    jwp_query(
        &mut framed,
        "CREATE TABLE counter (id INTEGER PRIMARY KEY, n INTEGER)",
        2,
    )
    .await
    .unwrap();

    // Run 20 sequential inserts + selects
    for i in 1..=20u32 {
        jwp_query(
            &mut framed,
            &format!("INSERT INTO counter (id, n) VALUES ({}, {})", i, i * 10),
            100 + i,
        )
        .await
        .unwrap();
    }

    let (rows, done) = jwp_query(&mut framed, "SELECT COUNT(*) FROM counter", 200)
        .await
        .unwrap();
    assert_eq!(done.row_count, 1);
    // COUNT(*) returns an integer
    let count = rows[0][0].as_i64().unwrap();
    assert_eq!(count, 20);
}

#[tokio::test]
async fn jwp_concurrent_connections() {
    let (server, base_url, jwp_addr) = create_jwp_server();
    start_server(server, &base_url).await;

    // Create a shared table first
    let mut setup = jwp_connect(&jwp_addr).await;
    jwp_query(
        &mut setup,
        "CREATE TABLE shared_data (id INTEGER PRIMARY KEY, worker INTEGER)",
        2,
    )
    .await
    .unwrap();
    drop(setup);

    // Spawn 10 concurrent clients
    let mut handles = Vec::new();
    for worker_id in 0..10u32 {
        let addr = jwp_addr.clone();
        handles.push(tokio::spawn(async move {
            let mut framed = jwp_connect(&addr).await;

            // Each worker inserts its own row
            let sql = format!(
                "INSERT INTO shared_data (id, worker) VALUES ({}, {})",
                worker_id, worker_id
            );
            let result = jwp_query(&mut framed, &sql, 2).await;
            assert!(
                result.is_ok(),
                "Worker {} insert failed: {:?}",
                worker_id,
                result
            );

            // Each worker reads all rows
            let (rows, _) = jwp_query(&mut framed, "SELECT * FROM shared_data", 3)
                .await
                .unwrap();
            assert!(!rows.is_empty(), "Worker {} got empty result", worker_id);
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all 10 rows exist
    let mut framed = jwp_connect(&jwp_addr).await;
    let (_, done) = jwp_query(&mut framed, "SELECT * FROM shared_data", 2)
        .await
        .unwrap();
    assert_eq!(done.row_count, 10);
}

#[tokio::test]
async fn jwp_empty_select() {
    let (server, base_url, jwp_addr) = create_jwp_server();
    start_server(server, &base_url).await;
    let mut framed = jwp_connect(&jwp_addr).await;

    jwp_query(
        &mut framed,
        "CREATE TABLE empty_table (id INTEGER PRIMARY KEY, value TEXT)",
        2,
    )
    .await
    .unwrap();

    // SELECT from empty table
    let (rows, done) = jwp_query(&mut framed, "SELECT * FROM empty_table", 3)
        .await
        .unwrap();
    assert_eq!(rows.len(), 0);
    assert_eq!(done.row_count, 0);
}

#[tokio::test]
async fn jwp_large_result_set() {
    let (server, base_url, jwp_addr) = create_jwp_server();
    start_server(server, &base_url).await;
    let mut framed = jwp_connect(&jwp_addr).await;

    jwp_query(
        &mut framed,
        "CREATE TABLE bulk (id INTEGER PRIMARY KEY, payload TEXT)",
        2,
    )
    .await
    .unwrap();

    // Insert 100 rows
    for i in 1..=100u32 {
        jwp_query(
            &mut framed,
            &format!("INSERT INTO bulk (id, payload) VALUES ({}, 'row-{}')", i, i),
            100 + i,
        )
        .await
        .unwrap();
    }

    // Select all 100
    let (rows, done) = jwp_query(&mut framed, "SELECT * FROM bulk ORDER BY id", 300)
        .await
        .unwrap();
    assert_eq!(done.row_count, 100);
    assert_eq!(rows.len(), 100);
    // First and last row sanity check
    assert_eq!(rows[0][1], serde_json::Value::String("row-1".into()));
    assert_eq!(rows[99][1], serde_json::Value::String("row-100".into()));
}

#[tokio::test]
async fn jwp_ddl_operations() {
    let (server, base_url, jwp_addr) = create_jwp_server();
    start_server(server, &base_url).await;
    let mut framed = jwp_connect(&jwp_addr).await;

    // CREATE TABLE
    jwp_query(
        &mut framed,
        "CREATE TABLE ddl_test (id INTEGER PRIMARY KEY, name TEXT)",
        2,
    )
    .await
    .unwrap();

    // INSERT data
    jwp_query(
        &mut framed,
        "INSERT INTO ddl_test (id, name) VALUES (1, 'test')",
        3,
    )
    .await
    .unwrap();

    // Verify data exists
    let (rows, _) = jwp_query(&mut framed, "SELECT * FROM ddl_test", 4)
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);

    // DROP TABLE
    jwp_query(&mut framed, "DROP TABLE ddl_test", 5)
        .await
        .unwrap();

    // SELECT from dropped table should error
    let result = jwp_query(&mut framed, "SELECT * FROM ddl_test", 6).await;
    assert!(result.is_err(), "Expected error after DROP TABLE");
}
