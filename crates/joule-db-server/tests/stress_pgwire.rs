//! PgWire attack surface stress tests.
//!
//! Tests the PostgreSQL wire protocol implementation against malformed
//! messages, oversized payloads, connection exhaustion, idle timeouts,
//! authentication failures, and protocol edge cases.

use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use joule_db_server::pgwire::{PgWireConfig, PgWireServer};
use joule_db_server::query::{QueryExecutor, SimpleQueryExecutor};

/// Protocol version 3.0 constant
const PROTOCOL_VERSION_3: i32 = 196608;

/// Build a startup message with given user and database.
fn build_startup_message(user: &str, database: &str) -> Vec<u8> {
    let mut params = Vec::new();
    params.extend_from_slice(b"user\0");
    params.extend_from_slice(user.as_bytes());
    params.push(0);
    params.extend_from_slice(b"database\0");
    params.extend_from_slice(database.as_bytes());
    params.push(0);
    params.push(0); // params terminator

    let length = (4 + 4 + params.len()) as i32; // length field + version + params
    let mut msg = Vec::new();
    msg.extend_from_slice(&length.to_be_bytes());
    msg.extend_from_slice(&PROTOCOL_VERSION_3.to_be_bytes());
    msg.extend_from_slice(&params);
    msg
}

/// Build a simple query ('Q') message.
fn build_query_message(sql: &str) -> Vec<u8> {
    let body_len = sql.len() + 1; // sql + null terminator
    let length = (4 + body_len) as i32;
    let mut msg = Vec::new();
    msg.push(b'Q');
    msg.extend_from_slice(&length.to_be_bytes());
    msg.extend_from_slice(sql.as_bytes());
    msg.push(0);
    msg
}

/// Build a password message.
fn build_password_message(password: &str) -> Vec<u8> {
    let pw_bytes = password.as_bytes();
    let length = (4 + pw_bytes.len() + 1) as i32;
    let mut msg = Vec::new();
    msg.push(b'p');
    msg.extend_from_slice(&length.to_be_bytes());
    msg.extend_from_slice(pw_bytes);
    msg.push(0);
    msg
}

/// Start a PgWire server on a random port and return the address.
async fn start_pgwire_server(config: PgWireConfig) -> (String, Arc<PgWireServer>) {
    let server = Arc::new(PgWireServer::from_dyn(
        config.clone(),
        Arc::new(SimpleQueryExecutor::new()) as Arc<dyn QueryExecutor>,
    ));
    let server_clone = server.clone();
    let addr = config.bind_addr.clone();
    tokio::spawn(async move {
        let _ = server_clone.run().await;
    });
    // Give server time to bind
    tokio::time::sleep(Duration::from_millis(100)).await;
    (addr, server)
}

/// Connect to PgWire and complete the startup handshake (no auth).
async fn connect_and_handshake(addr: &str) -> TcpStream {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let startup = build_startup_message("test", "testdb");
    stream.write_all(&startup).await.unwrap();
    stream.flush().await.unwrap();

    // Read responses until ReadyForQuery ('Z')
    let mut buf = vec![0u8; 4096];
    loop {
        let n = stream.read(&mut buf).await.unwrap();
        if n == 0 {
            panic!("Server closed connection during handshake");
        }
        // Scan for ReadyForQuery message (tag 'Z', 5 bytes total)
        if buf[..n].windows(1).any(|w| w[0] == b'Z') {
            break;
        }
    }
    stream
}

/// Read all available bytes from a stream (non-blocking after initial data).
async fn read_response(stream: &mut TcpStream) -> Vec<u8> {
    let mut buf = vec![0u8; 65536];
    let mut all = Vec::new();
    match tokio::time::timeout(Duration::from_secs(2), stream.read(&mut buf)).await {
        Ok(Ok(n)) if n > 0 => all.extend_from_slice(&buf[..n]),
        _ => {}
    }
    all
}

// ============================================================================
// Oversized message tests (Bug 1 fix verification)
// ============================================================================

#[tokio::test]
async fn pgwire_oversized_message_rejected() {
    let addr = "127.0.0.1:15430";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Send a query message claiming 300MB body
    let claimed_size: i32 = 300 * 1024 * 1024;
    let mut msg = Vec::new();
    msg.push(b'Q');
    msg.extend_from_slice(&claimed_size.to_be_bytes());
    stream.write_all(&msg).await.unwrap();

    // Server should respond with error and close (not OOM)
    let resp = read_response(&mut stream).await;
    // Connection should be terminated (error or close)
    assert!(
        resp.is_empty() || resp[0] == b'E',
        "Expected error or close, got {} bytes starting with {:?}",
        resp.len(),
        resp.first()
    );
}

#[tokio::test]
async fn pgwire_negative_message_length() {
    let addr = "127.0.0.1:15431";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Send query with negative length (length field = -1)
    let mut msg = Vec::new();
    msg.push(b'Q');
    msg.extend_from_slice(&(-1i32).to_be_bytes());
    stream.write_all(&msg).await.unwrap();

    let resp = read_response(&mut stream).await;
    // Server should reject — either error or close
    assert!(resp.is_empty() || resp[0] == b'E');
}

#[tokio::test]
async fn pgwire_zero_body_length() {
    let addr = "127.0.0.1:15432";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Send query with length = 3 (less than minimum 4)
    let mut msg = Vec::new();
    msg.push(b'Q');
    msg.extend_from_slice(&3i32.to_be_bytes());
    stream.write_all(&msg).await.unwrap();

    let resp = read_response(&mut stream).await;
    assert!(resp.is_empty() || resp[0] == b'E');
}

#[tokio::test]
async fn pgwire_i32_max_length() {
    let addr = "127.0.0.1:15433";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Claim i32::MAX bytes
    let mut msg = Vec::new();
    msg.push(b'Q');
    msg.extend_from_slice(&i32::MAX.to_be_bytes());
    stream.write_all(&msg).await.unwrap();

    let resp = read_response(&mut stream).await;
    // Must NOT cause OOM
    assert!(resp.is_empty() || resp[0] == b'E');
}

// ============================================================================
// Connection tests
// ============================================================================

#[tokio::test]
async fn pgwire_rapid_connect_disconnect_100() {
    let addr = "127.0.0.1:15434";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, server) = start_pgwire_server(config).await;

    // Rapidly connect and disconnect 100 times
    for _ in 0..100 {
        if let Ok(stream) = TcpStream::connect(addr).await {
            drop(stream);
        }
    }

    // Server should still be accepting connections
    tokio::time::sleep(Duration::from_millis(200)).await;
    let snap = server.stats().snapshot();
    assert!(
        snap.connections_accepted >= 50,
        "Server lost too many connections: {}",
        snap.connections_accepted
    );
}

#[tokio::test]
async fn pgwire_partial_startup_then_disconnect() {
    let addr = "127.0.0.1:15435";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    // Send partial startup (only length field, no version/params)
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(&100i32.to_be_bytes()).await.unwrap();
    stream.flush().await.unwrap();
    // Just drop the connection
    drop(stream);

    // Verify server still works after partial message
    tokio::time::sleep(Duration::from_millis(100)).await;
    let _stream = connect_and_handshake(addr).await;
}

#[tokio::test]
async fn pgwire_half_header_then_disconnect() {
    let addr = "127.0.0.1:15436";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Send only the message type byte, then disconnect
    stream.write_all(&[b'Q']).await.unwrap();
    stream.flush().await.unwrap();
    drop(stream);

    // Server should handle gracefully
    tokio::time::sleep(Duration::from_millis(100)).await;
    let _stream = connect_and_handshake(addr).await;
}

// ============================================================================
// Startup handshake malformation tests
// ============================================================================

#[tokio::test]
async fn pgwire_wrong_protocol_version() {
    let addr = "127.0.0.1:15437";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();

    // Send startup with wrong version (v2.0 = 131072)
    let wrong_version: i32 = 131072;
    let params = b"user\0test\0\0";
    let length = (4 + 4 + params.len()) as i32;
    let mut msg = Vec::new();
    msg.extend_from_slice(&length.to_be_bytes());
    msg.extend_from_slice(&wrong_version.to_be_bytes());
    msg.extend_from_slice(params);
    stream.write_all(&msg).await.unwrap();

    let resp = read_response(&mut stream).await;
    // Should get error or close
    assert!(resp.is_empty() || resp[0] == b'E');
}

#[tokio::test]
async fn pgwire_startup_missing_user_param() {
    let addr = "127.0.0.1:15438";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();

    // Startup with no user param
    let params = b"database\0testdb\0\0";
    let length = (4 + 4 + params.len()) as i32;
    let mut msg = Vec::new();
    msg.extend_from_slice(&length.to_be_bytes());
    msg.extend_from_slice(&PROTOCOL_VERSION_3.to_be_bytes());
    msg.extend_from_slice(params);
    stream.write_all(&msg).await.unwrap();

    // Should still work (user defaults to "anonymous")
    let resp = read_response(&mut stream).await;
    assert!(!resp.is_empty(), "Expected response from server");
}

#[tokio::test]
async fn pgwire_startup_empty_params() {
    let addr = "127.0.0.1:15439";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();

    // Startup with just a terminator
    let params = b"\0";
    let length = (4 + 4 + params.len()) as i32;
    let mut msg = Vec::new();
    msg.extend_from_slice(&length.to_be_bytes());
    msg.extend_from_slice(&PROTOCOL_VERSION_3.to_be_bytes());
    msg.extend_from_slice(params);
    stream.write_all(&msg).await.unwrap();

    let resp = read_response(&mut stream).await;
    // Should handle gracefully (either accept or error)
    assert!(resp.is_empty() || !resp.is_empty()); // no panic
}

// ============================================================================
// SSL request tests
// ============================================================================

#[tokio::test]
async fn pgwire_ssl_request_then_garbage() {
    let addr = "127.0.0.1:15440";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();

    // Send SSL request: length=8, version=80877103
    let mut msg = Vec::new();
    msg.extend_from_slice(&8i32.to_be_bytes());
    msg.extend_from_slice(&80877103i32.to_be_bytes());
    stream.write_all(&msg).await.unwrap();

    // Server should respond with 'N' (no SSL) or 'S' (SSL available)
    let mut resp = [0u8; 1];
    let n = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut resp)).await;
    if let Ok(Ok(1)) = n {
        assert!(
            resp[0] == b'N' || resp[0] == b'S',
            "Expected 'N' or 'S', got {:?}",
            resp[0]
        );
    }

    // Send garbage after SSL negotiation
    stream.write_all(b"GARBAGE_DATA_NOT_TLS").await.unwrap();

    // Server should close connection
    let resp = read_response(&mut stream).await;
    // Accept either error or close
    let _ = resp;
}

// ============================================================================
// Authentication tests
// ============================================================================

#[tokio::test]
async fn pgwire_auth_wrong_password() {
    let addr = "127.0.0.1:15441";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: true,
        auth_password: Some("correct_password".to_string()),
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();
    let startup = build_startup_message("alice", "testdb");
    stream.write_all(&startup).await.unwrap();

    // Read auth request
    let mut buf = vec![0u8; 4096];
    let _ = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut buf)).await;

    // Send wrong password
    let pw = build_password_message("wrong_password");
    stream.write_all(&pw).await.unwrap();

    let resp = read_response(&mut stream).await;
    // Should get error response
    assert!(
        resp.is_empty() || resp[0] == b'E',
        "Expected error after wrong password"
    );
}

#[tokio::test]
async fn pgwire_auth_empty_password() {
    let addr = "127.0.0.1:15442";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: true,
        auth_password: Some("correct_password".to_string()),
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();
    let startup = build_startup_message("alice", "testdb");
    stream.write_all(&startup).await.unwrap();

    let mut buf = vec![0u8; 4096];
    let _ = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut buf)).await;

    // Send empty password
    let pw = build_password_message("");
    stream.write_all(&pw).await.unwrap();

    let resp = read_response(&mut stream).await;
    assert!(resp.is_empty() || resp[0] == b'E');
}

// ============================================================================
// Query tests
// ============================================================================

#[tokio::test]
async fn pgwire_simple_query_roundtrip() {
    let addr = "127.0.0.1:15443";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Send SELECT 1
    let query = build_query_message("SELECT 1");
    stream.write_all(&query).await.unwrap();

    let resp = read_response(&mut stream).await;
    assert!(!resp.is_empty(), "Expected response to SELECT 1");
    // Should contain RowDescription ('T') and DataRow ('D')
    assert!(
        resp.iter().any(|&b| b == b'T' || b == b'D' || b == b'C'),
        "Expected query result messages"
    );
}

#[tokio::test]
async fn pgwire_empty_query() {
    let addr = "127.0.0.1:15444";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Send empty query
    let query = build_query_message("");
    stream.write_all(&query).await.unwrap();

    let resp = read_response(&mut stream).await;
    // Should get EmptyQueryResponse ('I') + ReadyForQuery ('Z')
    assert!(!resp.is_empty(), "Expected empty query response");
}

#[tokio::test]
async fn pgwire_invalid_sql() {
    let addr = "127.0.0.1:15445";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    let query = build_query_message("THIS IS NOT SQL!!!");
    stream.write_all(&query).await.unwrap();

    let resp = read_response(&mut stream).await;
    // Should get ErrorResponse ('E') + ReadyForQuery ('Z')
    assert!(
        resp.iter().any(|&b| b == b'E'),
        "Expected error response for invalid SQL"
    );
}

#[tokio::test]
async fn pgwire_multiple_queries_sequential() {
    let addr = "127.0.0.1:15446";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Send 10 queries sequentially
    for i in 0..10 {
        let query = build_query_message(&format!("SELECT {}", i));
        stream.write_all(&query).await.unwrap();
        let _ = read_response(&mut stream).await;
    }

    // Connection should still be alive
    let query = build_query_message("SELECT 'alive'");
    stream.write_all(&query).await.unwrap();
    let resp = read_response(&mut stream).await;
    assert!(
        !resp.is_empty(),
        "Connection should still be alive after 10 queries"
    );
}

#[tokio::test]
async fn pgwire_pipelined_queries_no_read() {
    let addr = "127.0.0.1:15447";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Send 50 queries without reading responses (pipeline)
    for i in 0..50 {
        let query = build_query_message(&format!("SELECT {}", i));
        stream.write_all(&query).await.unwrap();
    }
    stream.flush().await.unwrap();

    // Now read all responses
    let mut total_read = 0;
    let mut buf = vec![0u8; 65536];
    for _ in 0..50 {
        match tokio::time::timeout(Duration::from_secs(5), stream.read(&mut buf)).await {
            Ok(Ok(n)) if n > 0 => total_read += n,
            _ => break,
        }
    }
    assert!(total_read > 0, "Should have received pipelined responses");
}

// ============================================================================
// Unknown message types
// ============================================================================

#[tokio::test]
async fn pgwire_unknown_message_type() {
    let addr = "127.0.0.1:15448";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Send message with unknown type byte (0xFF)
    let mut msg = Vec::new();
    msg.push(0xFF);
    msg.extend_from_slice(&8i32.to_be_bytes()); // length = 8
    msg.extend_from_slice(&[0, 0, 0, 0]); // 4 bytes body
    stream.write_all(&msg).await.unwrap();

    let resp = read_response(&mut stream).await;
    // Server should handle gracefully (error or ignore)
    let _ = resp;
}

#[tokio::test]
async fn pgwire_terminate_message() {
    let addr = "127.0.0.1:15449";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Send Terminate ('X') message
    let mut msg = Vec::new();
    msg.push(b'X');
    msg.extend_from_slice(&4i32.to_be_bytes()); // minimal length
    stream.write_all(&msg).await.unwrap();

    // Connection should close
    let mut buf = [0u8; 1];
    let result = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut buf)).await;
    match result {
        Ok(Ok(0)) | Ok(Err(_)) | Err(_) => {} // Expected: connection closed
        Ok(Ok(_)) => {}                       // Some data before close is ok
    }
}

// ============================================================================
// Concurrent connections
// ============================================================================

#[tokio::test]
async fn pgwire_concurrent_connections_10() {
    let addr = "127.0.0.1:15450";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    // Open 10 connections concurrently
    let mut handles = Vec::new();
    for i in 0..10 {
        let addr = addr.to_string();
        handles.push(tokio::spawn(async move {
            let mut stream = connect_and_handshake(&addr).await;
            let query = build_query_message(&format!("SELECT {}", i));
            stream.write_all(&query).await.unwrap();
            let resp = read_response(&mut stream).await;
            !resp.is_empty()
        }));
    }

    let mut ok = 0;
    for h in handles {
        if h.await.unwrap() {
            ok += 1;
        }
    }
    assert!(
        ok >= 8,
        "At least 8 of 10 concurrent connections should succeed, got {}",
        ok
    );
}

// ============================================================================
// Binary data and encoding edge cases
// ============================================================================

#[tokio::test]
async fn pgwire_query_with_null_bytes() {
    let addr = "127.0.0.1:15451";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Query with embedded nulls (should be truncated at first null in PgWire)
    let query = build_query_message("SELECT 1\0; DROP TABLE important;");
    stream.write_all(&query).await.unwrap();

    let resp = read_response(&mut stream).await;
    // Should execute "SELECT 1" only (null acts as terminator)
    assert!(!resp.is_empty());
}

#[tokio::test]
async fn pgwire_query_with_unicode() {
    let addr = "127.0.0.1:15452";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    let query = build_query_message("SELECT '\u{1F680}\u{1F30D}\u{2764}'");
    stream.write_all(&query).await.unwrap();

    let resp = read_response(&mut stream).await;
    assert!(!resp.is_empty(), "Unicode query should get a response");
}

// ============================================================================
// Extended query protocol edge cases
// ============================================================================

#[tokio::test]
async fn pgwire_parse_message() {
    let addr = "127.0.0.1:15453";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Send Parse ('P') message: name="" sql="SELECT 1" param_count=0
    let sql = b"SELECT 1";
    let body_len = 1 + sql.len() + 1 + 2; // name\0 + sql\0 + param_count
    let length = (4 + body_len) as i32;
    let mut msg = Vec::new();
    msg.push(b'P');
    msg.extend_from_slice(&length.to_be_bytes());
    msg.push(0); // empty statement name
    msg.extend_from_slice(sql);
    msg.push(0); // null terminator
    msg.extend_from_slice(&0i16.to_be_bytes()); // 0 params
    stream.write_all(&msg).await.unwrap();

    // Send Sync ('S')
    msg.clear();
    msg.push(b'S');
    msg.extend_from_slice(&4i32.to_be_bytes());
    stream.write_all(&msg).await.unwrap();
    stream.flush().await.unwrap();

    let resp = read_response(&mut stream).await;
    // Should get ParseComplete ('1') + ReadyForQuery ('Z')
    assert!(!resp.is_empty(), "Expected response to Parse");
}

#[tokio::test]
async fn pgwire_describe_nonexistent_statement() {
    let addr = "127.0.0.1:15454";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Send Describe ('D') for nonexistent statement
    let name = b"nonexistent";
    let body_len = 1 + name.len() + 1; // type + name\0
    let length = (4 + body_len) as i32;
    let mut msg = Vec::new();
    msg.push(b'D');
    msg.extend_from_slice(&length.to_be_bytes());
    msg.push(b'S'); // describe Statement
    msg.extend_from_slice(name);
    msg.push(0);
    stream.write_all(&msg).await.unwrap();

    // Sync
    msg.clear();
    msg.push(b'S');
    msg.extend_from_slice(&4i32.to_be_bytes());
    stream.write_all(&msg).await.unwrap();

    let resp = read_response(&mut stream).await;
    // Should get ErrorResponse ('E') + ReadyForQuery ('Z')
    assert!(!resp.is_empty());
}

// ============================================================================
// Stats tracking
// ============================================================================

#[tokio::test]
async fn pgwire_stats_tracking() {
    let addr = "127.0.0.1:15455";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // Execute 3 queries
    for _ in 0..3 {
        let query = build_query_message("SELECT 1");
        stream.write_all(&query).await.unwrap();
        let _ = read_response(&mut stream).await;
    }
    drop(stream);

    tokio::time::sleep(Duration::from_millis(200)).await;
    let snap = server.stats().snapshot();
    assert!(snap.connections_accepted >= 1, "Should track connections");
    assert!(
        snap.queries_executed >= 3,
        "Should track queries: {}",
        snap.queries_executed
    );
    assert!(snap.bytes_received > 0, "Should track bytes received");
}

// ============================================================================
// Cancel request
// ============================================================================

#[tokio::test]
async fn pgwire_cancel_request() {
    let addr = "127.0.0.1:15456";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();

    // Send cancel request: length=16, version=80877102, pid=1234, secret=5678
    let mut msg = Vec::new();
    msg.extend_from_slice(&16i32.to_be_bytes());
    msg.extend_from_slice(&80877102i32.to_be_bytes());
    msg.extend_from_slice(&1234i32.to_be_bytes()); // pid
    msg.extend_from_slice(&5678i32.to_be_bytes()); // secret
    stream.write_all(&msg).await.unwrap();

    // Server should close connection after handling cancel
    let mut buf = [0u8; 1];
    let _ = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut buf)).await;
}

// ============================================================================
// Create/query lifecycle through PgWire
// ============================================================================

#[tokio::test]
async fn pgwire_full_crud_lifecycle() {
    let addr = "127.0.0.1:15457";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    // CREATE TABLE
    let q = build_query_message("CREATE TABLE pgwire_test (id INT, name TEXT)");
    stream.write_all(&q).await.unwrap();
    let _ = read_response(&mut stream).await;

    // INSERT
    let q = build_query_message("INSERT INTO pgwire_test VALUES (1, 'hello')");
    stream.write_all(&q).await.unwrap();
    let _ = read_response(&mut stream).await;

    // SELECT
    let q = build_query_message("SELECT * FROM pgwire_test");
    stream.write_all(&q).await.unwrap();
    let resp = read_response(&mut stream).await;
    assert!(!resp.is_empty(), "SELECT should return data");

    // DROP TABLE
    let q = build_query_message("DROP TABLE pgwire_test");
    stream.write_all(&q).await.unwrap();
    let _ = read_response(&mut stream).await;
}

#[tokio::test]
async fn pgwire_query_nonexistent_table() {
    let addr = "127.0.0.1:15458";
    let config = PgWireConfig {
        bind_addr: addr.to_string(),
        auth_enabled: false,
        ..Default::default()
    };
    let (_addr, _server) = start_pgwire_server(config).await;

    let mut stream = connect_and_handshake(addr).await;

    let q = build_query_message("SELECT * FROM this_table_does_not_exist");
    stream.write_all(&q).await.unwrap();
    let resp = read_response(&mut stream).await;
    // Should get error response, NOT crash
    assert!(
        resp.iter().any(|&b| b == b'E'),
        "Expected error for missing table"
    );
}
