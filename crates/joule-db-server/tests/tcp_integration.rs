//! TCP Integration Tests
//!
//! These tests verify the server works end-to-end over actual TCP connections.
//! Unlike unit tests that use tower::ServiceExt::oneshot, these tests start
//! a real HTTP server and make actual network requests.

use joule_db_server::{Server, ServerConfig};
use std::net::TcpListener;
use std::time::Duration;

/// Find an available port by binding to port 0
fn find_available_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Create a test server on a random port and return (server, base_url)
fn create_test_server() -> (Server, String) {
    let port = find_available_port();
    let tcp_port = find_available_port();
    let dir = tempfile::tempdir().unwrap();
    let config = ServerConfig {
        http_addr: format!("127.0.0.1:{}", port),
        tcp_addr: format!("127.0.0.1:{}", tcp_port),
        db_path: dir.path().to_string_lossy().to_string(),
        enable_websocket: false,
        enable_tcp: false, // Disable for HTTP-only tests
        max_tcp_connections: 100,
        enable_webtransport: false,
        webtransport_port: 0,
        enable_pgwire: false,
        pgwire_addr: "127.0.0.1:0".to_string(),
        enable_jwp: false,
        jwp_addr: "127.0.0.1:0".to_string(),
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
    // Keep dir alive by leaking it (for tests)
    std::mem::forget(dir);
    let server = Server::new(config).unwrap();
    let base_url = format!("http://127.0.0.1:{}", port);
    (server, base_url)
}

/// Start server in background and wait for it to be ready
async fn start_server_background(server: Server, base_url: &str) {
    let url = base_url.to_string();

    // Spawn server in background task
    tokio::spawn(async move {
        server.run().await.unwrap();
    });

    // Wait for server to be ready (poll health endpoint)
    let client = reqwest::Client::new();
    for _ in 0..50 {
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

#[tokio::test]
async fn test_tcp_health_check() {
    let (server, base_url) = create_test_server();
    start_server_background(server, &base_url).await;

    let client = reqwest::Client::new();
    let response = client
        .get(format!("{}/health", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    // Health endpoint returns JSON with status field
    let json: serde_json::Value = response.json().await.unwrap();
    assert!(json.get("status").is_some());
}

#[tokio::test]
async fn test_tcp_put_get_delete_roundtrip() {
    let (server, base_url) = create_test_server();
    start_server_background(server, &base_url).await;

    let client = reqwest::Client::new();

    // PUT a key
    let put_response = client
        .post(format!("{}/api/v1/keys/testkey", base_url))
        .body("hello world")
        .send()
        .await
        .unwrap();
    assert_eq!(put_response.status(), reqwest::StatusCode::OK);

    // GET the key back
    let get_response = client
        .get(format!("{}/api/v1/keys/testkey", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(get_response.status(), reqwest::StatusCode::OK);
    let value: Option<String> = get_response.json().await.unwrap();
    assert_eq!(value, Some("hello world".to_string()));

    // DELETE the key
    let delete_response = client
        .delete(format!("{}/api/v1/keys/testkey", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(delete_response.status(), reqwest::StatusCode::OK);
    let deleted: bool = delete_response.json().await.unwrap();
    assert!(deleted);

    // GET should return None now
    let get_response = client
        .get(format!("{}/api/v1/keys/testkey", base_url))
        .send()
        .await
        .unwrap();
    let value: Option<String> = get_response.json().await.unwrap();
    assert_eq!(value, None);
}

#[tokio::test]
async fn test_tcp_multiple_keys() {
    let (server, base_url) = create_test_server();
    start_server_background(server, &base_url).await;

    let client = reqwest::Client::new();

    // Insert 100 keys
    for i in 0..100 {
        let response = client
            .post(format!("{}/api/v1/keys/key{:03}", base_url, i))
            .body(format!("value{:03}", i))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
    }

    // Verify all 100 keys
    for i in 0..100 {
        let response = client
            .get(format!("{}/api/v1/keys/key{:03}", base_url, i))
            .send()
            .await
            .unwrap();
        let value: Option<String> = response.json().await.unwrap();
        assert_eq!(value, Some(format!("value{:03}", i)));
    }
}

#[tokio::test]
async fn test_tcp_binary_values() {
    let (server, base_url) = create_test_server();
    start_server_background(server, &base_url).await;

    let client = reqwest::Client::new();

    // PUT binary data (will fail to return as string but we test the PUT works)
    let binary_data: Vec<u8> = (0..255).collect();
    let response = client
        .post(format!("{}/api/v1/keys/binkey", base_url))
        .body(binary_data)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK);
}

#[tokio::test]
async fn test_tcp_concurrent_requests() {
    let (server, base_url) = create_test_server();
    start_server_background(server, &base_url).await;

    let client = reqwest::Client::new();

    // First seed some data
    for i in 0..10 {
        client
            .post(format!("{}/api/v1/keys/concurrent{}", base_url, i))
            .body(format!("value{}", i))
            .send()
            .await
            .unwrap();
    }

    // Fire off many concurrent GET requests
    let mut handles = Vec::new();
    for i in 0..100 {
        let client = client.clone();
        let url = format!("{}/api/v1/keys/concurrent{}", base_url, i % 10);
        handles.push(tokio::spawn(async move {
            let response = client.get(&url).send().await.unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            let value: Option<String> = response.json().await.unwrap();
            assert!(value.is_some());
        }));
    }

    // Wait for all requests to complete
    for handle in handles {
        handle.await.unwrap();
    }
}

#[tokio::test]
async fn test_tcp_overwrite_key() {
    let (server, base_url) = create_test_server();
    start_server_background(server, &base_url).await;

    let client = reqwest::Client::new();

    // PUT initial value
    client
        .post(format!("{}/api/v1/keys/overwrite", base_url))
        .body("initial")
        .send()
        .await
        .unwrap();

    // Overwrite with new value
    client
        .post(format!("{}/api/v1/keys/overwrite", base_url))
        .body("updated")
        .send()
        .await
        .unwrap();

    // Verify new value
    let response = client
        .get(format!("{}/api/v1/keys/overwrite", base_url))
        .send()
        .await
        .unwrap();
    let value: Option<String> = response.json().await.unwrap();
    assert_eq!(value, Some("updated".to_string()));
}

#[tokio::test]
async fn test_tcp_get_nonexistent() {
    let (server, base_url) = create_test_server();
    start_server_background(server, &base_url).await;

    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/api/v1/keys/doesnotexist", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let value: Option<String> = response.json().await.unwrap();
    assert_eq!(value, None);
}

#[tokio::test]
async fn test_tcp_delete_nonexistent() {
    let (server, base_url) = create_test_server();
    start_server_background(server, &base_url).await;

    let client = reqwest::Client::new();

    let response = client
        .delete(format!("{}/api/v1/keys/doesnotexist", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let deleted: bool = response.json().await.unwrap();
    assert!(!deleted);
}

#[tokio::test]
async fn test_tcp_large_value() {
    let (server, base_url) = create_test_server();
    start_server_background(server, &base_url).await;

    let client = reqwest::Client::new();

    // Create a 10KB value (within page size limits)
    let large_value = "x".repeat(10 * 1024);

    let response = client
        .post(format!("{}/api/v1/keys/largekey", base_url))
        .body(large_value.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK);

    // Verify we can get it back
    let response = client
        .get(format!("{}/api/v1/keys/largekey", base_url))
        .send()
        .await
        .unwrap();
    let value: Option<String> = response.json().await.unwrap();
    assert_eq!(value, Some(large_value));
}
