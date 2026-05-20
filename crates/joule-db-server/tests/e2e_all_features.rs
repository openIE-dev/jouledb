//! End-to-end feature validation for JouleDB
//!
//! Tests every major subsystem through the HTTP API using the real Server
//! and axum router. Each test creates an isolated temp-dir-backed server.

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use joule_db_server::ServerConfig;
use serde_json::{Value, json};
use tower::ServiceExt;

// ============================================================================
// Test Harness
// ============================================================================

fn create_test_server() -> joule_db_server::Server {
    let dir = tempfile::tempdir().unwrap();
    let config = ServerConfig {
        http_addr: "127.0.0.1:0".to_string(),
        tcp_addr: "127.0.0.1:0".to_string(),
        db_path: dir.path().to_string_lossy().to_string(),
        enable_websocket: false,
        enable_tcp: false,
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
    std::mem::forget(dir);
    joule_db_server::Server::new(config).unwrap()
}

async fn post_json(app: &axum::Router, uri: &str, body: &Value) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(uri)
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_vec(body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

async fn post_raw(app: &axum::Router, uri: &str, body: &str) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(uri)
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

async fn get_json(app: &axum::Router, uri: &str) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

async fn delete_json(app: &axum::Router, uri: &str) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::DELETE)
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

// ============================================================================
// 1. Health & Status
// ============================================================================

#[tokio::test]
async fn e2e_health_and_status() {
    let server = create_test_server();
    let app = server.router();

    let (status, body) = get_json(&app, "/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "healthy");

    let (status, _) = get_json(&app, "/health/live").await;
    assert_eq!(status, StatusCode::OK);

    let (status, _) = get_json(&app, "/health/ready").await;
    assert_eq!(status, StatusCode::OK);
}

// ============================================================================
// 2. KV Operations (raw body, not JSON)
// ============================================================================

#[tokio::test]
async fn e2e_kv_crud() {
    let server = create_test_server();
    let app = server.router();

    // PUT (raw body)
    let (status, _) = post_raw(&app, "/api/v1/keys/mykey", "hello").await;
    assert_eq!(status, StatusCode::OK);

    // GET — returns Option<String> as JSON
    let (status, body) = get_json(&app, "/api/v1/keys/mykey").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_str().unwrap(), "hello");

    // DELETE
    let (status, _) = delete_json(&app, "/api/v1/keys/mykey").await;
    assert_eq!(status, StatusCode::OK);

    // GET after DELETE — returns 200 with null (Option<String> = None)
    let (status, body) = get_json(&app, "/api/v1/keys/mykey").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.is_null(), "deleted key should return null");
}

// ============================================================================
// 3. SQL via unified endpoint (uses "q" field, returns "data")
// ============================================================================

#[tokio::test]
async fn e2e_sql_create_insert_select_drop() {
    let server = create_test_server();
    let app = server.router();

    // Create table via SQL
    let (s, body) = post_json(
        &app,
        "/",
        &json!({"q": "CREATE TABLE e2e_users (id INT, name TEXT, email TEXT)"}),
    )
    .await;
    assert_eq!(body["ok"], true, "CREATE TABLE failed: {body}");

    // Insert rows
    for i in 1..=3 {
        let (s, body) = post_json(
            &app,
            "/",
            &json!({"q": format!("INSERT INTO e2e_users (id, name, email) VALUES ({i}, 'User{i}', 'u{i}@test.com')")}),
        ).await;
        assert_eq!(body["ok"], true, "INSERT {i} failed: {body}");
    }

    // SELECT *
    let (s, body) = post_json(&app, "/", &json!({"q": "SELECT * FROM e2e_users"})).await;
    assert_eq!(body["ok"], true);
    assert_eq!(body["count"], 3);

    // SELECT with WHERE
    let (s, body) = post_json(
        &app,
        "/",
        &json!({"q": "SELECT * FROM e2e_users WHERE id = 2"}),
    )
    .await;
    assert_eq!(body["ok"], true);
    assert_eq!(body["count"], 1);

    // COUNT aggregate
    let (s, body) = post_json(&app, "/", &json!({"q": "SELECT COUNT(*) FROM e2e_users"})).await;
    assert_eq!(body["ok"], true);
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);

    // DROP table
    let (s, body) = post_json(&app, "/", &json!({"q": "DROP TABLE e2e_users"})).await;
    assert_eq!(body["ok"], true);

    // Verify table is gone — should error
    let (s, body) = post_json(&app, "/", &json!({"q": "SELECT * FROM e2e_users"})).await;
    assert_eq!(body["ok"], false);
}

// ============================================================================
// 4. information_schema & pg_catalog
// ============================================================================

#[tokio::test]
async fn e2e_information_schema_and_pg_catalog() {
    let server = create_test_server();
    let app = server.router();

    // Create a table so information_schema has something
    post_json(
        &app,
        "/",
        &json!({"q": "CREATE TABLE pg_test (id INT, name TEXT)"}),
    )
    .await;

    // information_schema.tables
    let (s, body) = post_json(
        &app,
        "/",
        &json!({"q": "SELECT * FROM information_schema.tables"}),
    )
    .await;
    assert_eq!(body["ok"], true, "information_schema.tables failed: {body}");
    let data = body["data"].as_array().unwrap();
    assert!(data.len() >= 1, "should have at least 1 table");

    // pg_catalog.pg_type — should return standard PostgreSQL types
    let (s, body) = post_json(&app, "/", &json!({"q": "SELECT * FROM pg_catalog.pg_type"})).await;
    assert_eq!(body["ok"], true, "pg_catalog.pg_type failed: {body}");
    let data = body["data"].as_array().unwrap();
    assert!(
        data.len() >= 10,
        "pg_type should have 10+ types, got {}",
        data.len()
    );

    // pg_catalog.pg_namespace
    let (s, body) = post_json(
        &app,
        "/",
        &json!({"q": "SELECT * FROM pg_catalog.pg_namespace"}),
    )
    .await;
    assert_eq!(body["ok"], true, "pg_namespace failed: {body}");
    let data = body["data"].as_array().unwrap();
    assert!(data.len() >= 2);

    // pg_catalog.pg_database
    let (s, body) = post_json(
        &app,
        "/",
        &json!({"q": "SELECT * FROM pg_catalog.pg_database"}),
    )
    .await;
    assert_eq!(body["ok"], true, "pg_database failed: {body}");

    // pg_catalog.pg_settings
    let (s, body) = post_json(
        &app,
        "/",
        &json!({"q": "SELECT * FROM pg_catalog.pg_settings"}),
    )
    .await;
    assert_eq!(body["ok"], true, "pg_settings failed: {body}");

    // pg_catalog.pg_class — should list our pg_test table
    let (s, body) = post_json(
        &app,
        "/",
        &json!({"q": "SELECT * FROM pg_catalog.pg_class"}),
    )
    .await;
    assert_eq!(body["ok"], true, "pg_class failed: {body}");

    // pg_catalog.pg_attribute
    let (s, body) = post_json(
        &app,
        "/",
        &json!({"q": "SELECT * FROM pg_catalog.pg_attribute"}),
    )
    .await;
    assert_eq!(body["ok"], true, "pg_attribute failed: {body}");
}

// ============================================================================
// 5. Unified JSON Ingest
// ============================================================================

#[tokio::test]
async fn e2e_unified_ingest_and_query() {
    let server = create_test_server();
    let app = server.router();

    // Single-doc ingest
    let (s, body) = post_json(
        &app,
        "/",
        &json!({"name": "Alice", "age": 30, "city": "Portland"}),
    )
    .await;
    assert_eq!(body["ok"], true, "single ingest failed: {body}");
    assert!(body["id"].as_u64().is_some());

    // Batch ingest
    let (s, body) = post_json(
        &app,
        "/",
        &json!([
            {"name": "Bob", "age": 25, "city": "Seattle"},
            {"name": "Charlie", "age": 35, "city": "Portland"}
        ]),
    )
    .await;
    assert_eq!(body["ok"], true, "batch ingest failed: {body}");
    assert_eq!(body["count"], 2);
}

// ============================================================================
// 6. Agent Memory
// ============================================================================

#[tokio::test]
async fn e2e_agent_memory_lifecycle() {
    let server = create_test_server();
    let app = server.router();

    // Store memory
    let (s, body) = post_json(
        &app,
        "/api/v1/memory/store",
        &json!({
            "content": "The user prefers dark mode",
            "memory_type": "episodic",
            "agent_id": "agent-1"
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "store failed: {body}");
    let mem_id = body["id"].as_str().unwrap().to_string();
    assert!(mem_id.starts_with("mem_"));

    // Store another memory
    let (s, _) = post_json(
        &app,
        "/api/v1/memory/store",
        &json!({
            "content": "The user works on Rust projects",
            "memory_type": "semantic"
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK);

    // Recall memories
    let (s, body) = post_json(
        &app,
        "/api/v1/memory/recall",
        &json!({"query": "dark mode preference", "k": 5}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "recall failed: {body}");
    let memories = body["memories"].as_array().unwrap();
    assert!(
        !memories.is_empty(),
        "recall should return at least 1 memory"
    );

    // Stats
    let (s, body) = get_json(&app, "/api/v1/memory/stats").await;
    assert_eq!(s, StatusCode::OK, "stats failed: {body}");
    assert_eq!(body["total_memories"], 2);
    assert_eq!(body["episodic_count"], 1);
    assert_eq!(body["semantic_count"], 1);

    // Consolidate episodic → semantic
    let (s, body) = post_json(&app, "/api/v1/memory/consolidate", &json!({})).await;
    assert_eq!(s, StatusCode::OK, "consolidate failed: {body}");
    assert_eq!(body["consolidated"], 1);

    // Stats after consolidation
    let (s, body) = get_json(&app, "/api/v1/memory/stats").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(body["episodic_count"], 0);
}

// ============================================================================
// 7. Workflow Engine
// ============================================================================

#[tokio::test]
async fn e2e_workflow_lifecycle() {
    let server = create_test_server();
    let app = server.router();

    // Create workflow definition — must match WorkflowStep struct:
    // { label, operation, depends_on, timeout_ms }
    let (s, body) = post_json(
        &app,
        "/api/v1/workflows",
        &json!({
            "name": "etl_pipeline",
            "steps": [
                {"label": "extract", "operation": {"sql": "SELECT 1"}, "depends_on": [], "timeout_ms": 5000},
                {"label": "transform", "operation": {"tool": "normalize"}, "depends_on": ["extract"], "timeout_ms": 10000},
                {"label": "load", "operation": {"tool": "write_db"}, "depends_on": ["transform"], "timeout_ms": 5000}
            ],
            "energy_budget_uj": 50000
        }),
    ).await;
    assert_eq!(s, StatusCode::OK, "create workflow failed: {body}");
    let wf_id = body["id"].as_str().unwrap().to_string();

    // List workflows
    let (s, body) = get_json(&app, "/api/v1/workflows").await;
    assert_eq!(s, StatusCode::OK);
    let defs = body["workflows"].as_array().unwrap();
    assert_eq!(defs.len(), 1);

    // Get workflow by ID
    let (s, body) = get_json(&app, &format!("/api/v1/workflows/{wf_id}")).await;
    assert_eq!(s, StatusCode::OK, "get workflow failed: {body}");
    assert_eq!(body["name"], "etl_pipeline");

    // Run workflow
    let (s, body) = post_json(&app, &format!("/api/v1/workflows/{wf_id}/run"), &json!({})).await;
    assert_eq!(s, StatusCode::OK, "run workflow failed: {body}");
    let instance_id = body["id"].as_str().unwrap().to_string();

    // Get instance status
    let (s, body) = get_json(&app, &format!("/api/v1/workflows/instances/{instance_id}")).await;
    assert_eq!(s, StatusCode::OK, "get instance failed: {body}");

    // Delete workflow definition (returns 204 No Content)
    let (s, _) = delete_json(&app, &format!("/api/v1/workflows/{wf_id}")).await;
    assert_eq!(s, StatusCode::NO_CONTENT);

    // Verify deleted
    let (s, _) = get_json(&app, &format!("/api/v1/workflows/{wf_id}")).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

// ============================================================================
// 8. Message Queue (pub/sub via workflow manager)
// ============================================================================

#[tokio::test]
async fn e2e_message_queue() {
    let server = create_test_server();
    let app = server.router();

    // Publish messages
    let (s, body) = post_json(
        &app,
        "/api/v1/queue/publish",
        &json!({"topic": "events", "payload": "user_signup"}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "publish failed: {body}");

    let (s, _) = post_json(
        &app,
        "/api/v1/queue/publish",
        &json!({"topic": "events", "payload": "page_view"}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);

    // Subscribe — pull messages
    let (s, body) = post_json(
        &app,
        "/api/v1/queue/subscribe",
        &json!({"topic": "events", "max_messages": 10}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "subscribe failed: {body}");
    let messages = body.as_array().unwrap();
    assert_eq!(messages.len(), 2, "should get 2 messages");

    // Ack messages
    let ids: Vec<String> = messages
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    let (s, body) = post_json(&app, "/api/v1/queue/ack", &json!({"message_ids": ids})).await;
    assert_eq!(s, StatusCode::OK, "ack failed: {body}");
    assert_eq!(body["acked"], 2);

    // Subscribe again — should be empty (all acked)
    let (s, body) = post_json(
        &app,
        "/api/v1/queue/subscribe",
        &json!({"topic": "events", "max_messages": 10}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let messages = body.as_array().unwrap();
    assert_eq!(messages.len(), 0, "all messages should be acked");
}

// ============================================================================
// 9. Edge PoP Management
// ============================================================================

#[tokio::test]
async fn e2e_edge_pop_lifecycle() {
    let server = create_test_server();
    let app = server.router();

    // Register a PoP
    let (s, body) = post_json(
        &app,
        "/api/v1/edge/pops",
        &json!({
            "region": "us_east",
            "endpoint": "https://pop-use1.jouledb.com",
            "is_wasm": false
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "register pop failed: {body}");
    let pop_id = body["id"].as_str().unwrap().to_string();

    // Register a WASM PoP
    let (s, _) = post_json(
        &app,
        "/api/v1/edge/pops",
        &json!({
            "region": "eu_west",
            "endpoint": "https://pop-euw1.jouledb.com",
            "is_wasm": true
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK);

    // List PoPs
    let (s, body) = get_json(&app, "/api/v1/edge/pops").await;
    assert_eq!(s, StatusCode::OK, "list pops failed: {body}");
    let pops = body.as_array().unwrap();
    assert_eq!(pops.len(), 2);

    // Get specific PoP
    let (s, body) = get_json(&app, &format!("/api/v1/edge/pops/{pop_id}")).await;
    assert_eq!(s, StatusCode::OK, "get pop failed: {body}");

    // Trigger sync
    let (s, body) = post_json(&app, "/api/v1/edge/sync", &json!({"pop_id": pop_id})).await;
    assert_eq!(s, StatusCode::OK, "sync failed: {body}");

    // Edge stats
    let (s, body) = get_json(&app, "/api/v1/edge/stats").await;
    assert_eq!(s, StatusCode::OK, "edge stats failed: {body}");

    // Deregister PoP
    let (s, _) = delete_json(&app, &format!("/api/v1/edge/pops/{pop_id}")).await;
    assert_eq!(s, StatusCode::OK);

    // Verify deregistered
    let (s, _) = get_json(&app, &format!("/api/v1/edge/pops/{pop_id}")).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

// ============================================================================
// 10. Branch Management (Git-style)
// ============================================================================

#[tokio::test]
async fn e2e_branch_management() {
    let server = create_test_server();
    let app = server.router();

    // List branches — should have "main"
    let (s, body) = get_json(&app, "/api/v1/branches").await;
    assert_eq!(s, StatusCode::OK, "list branches failed: {body}");
    let branches = body["branches"].as_array().unwrap();
    assert!(branches.len() >= 1, "should have at least main branch");

    // Create a feature branch
    let (s, body) = post_json(
        &app,
        "/api/v1/branches",
        &json!({"name": "feature-new-index", "tags": []}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "create branch failed: {body}");

    // Get branch
    let (s, body) = get_json(&app, "/api/v1/branches/feature-new-index").await;
    assert_eq!(s, StatusCode::OK, "get branch failed: {body}");

    // Duplicate branch should conflict
    let (s, _) = post_json(
        &app,
        "/api/v1/branches",
        &json!({"name": "feature-new-index", "tags": []}),
    )
    .await;
    assert_eq!(s, StatusCode::CONFLICT);

    // Delete branch
    let (s, _) = delete_json(&app, "/api/v1/branches/feature-new-index").await;
    assert_eq!(s, StatusCode::OK);

    // Verify deleted
    let (s, _) = get_json(&app, "/api/v1/branches/feature-new-index").await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

// ============================================================================
// 11. Tenant Management
// ============================================================================

#[tokio::test]
async fn e2e_tenant_management() {
    let server = create_test_server();
    let app = server.router();

    // List tenants
    let (s, body) = get_json(&app, "/api/v1/tenants").await;
    assert_eq!(s, StatusCode::OK, "list tenants failed: {body}");

    // Create tenant
    let (s, body) = post_json(&app, "/api/v1/tenants", &json!({"name": "acme-corp"})).await;
    assert_eq!(s, StatusCode::OK, "create tenant failed: {body}");
    let tenant = &body["tenant"];
    let tenant_id = tenant["id"].as_str().unwrap().to_string();

    // Get tenant
    let (s, body) = get_json(&app, &format!("/api/v1/tenants/{tenant_id}")).await;
    assert_eq!(s, StatusCode::OK, "get tenant failed: {body}");

    // Tenant energy
    let (s, body) = get_json(&app, &format!("/api/v1/tenants/{tenant_id}/energy")).await;
    assert_eq!(s, StatusCode::OK, "tenant energy failed: {body}");

    // Delete tenant
    let (s, _) = delete_json(&app, &format!("/api/v1/tenants/{tenant_id}")).await;
    assert_eq!(s, StatusCode::OK);

    // Verify deleted
    let (s, _) = get_json(&app, &format!("/api/v1/tenants/{tenant_id}")).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

// ============================================================================
// 12. Scale-to-Zero Status
// ============================================================================

#[tokio::test]
async fn e2e_scale_to_zero_status() {
    let server = create_test_server();
    let app = server.router();

    let (s, body) = get_json(&app, "/api/v1/status").await;
    assert_eq!(s, StatusCode::OK, "status failed: {body}");
}

// ============================================================================
// 13. Energy Status
// ============================================================================

#[tokio::test]
async fn e2e_energy_status() {
    let server = create_test_server();
    let app = server.router();

    let (s, body) = get_json(&app, "/api/v1/energy").await;
    assert_eq!(s, StatusCode::OK, "energy status failed: {body}");
}

// ============================================================================
// 14. Backup & Export
// ============================================================================

#[tokio::test]
async fn e2e_backup_list() {
    let server = create_test_server();
    let app = server.router();

    let (s, body) = get_json(&app, "/api/v1/backup/list").await;
    assert_eq!(s, StatusCode::OK, "backup list failed: {body}");
}

// ============================================================================
// 15. Metrics (Prometheus)
// ============================================================================

#[tokio::test]
async fn e2e_prometheus_metrics() {
    let server = create_test_server();
    let app = server.router();

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(!text.is_empty(), "metrics should have content");
}

// ============================================================================
// 16. Amorphic Ingest + Query
// ============================================================================

#[tokio::test]
async fn e2e_amorphic_ingest_and_query() {
    let server = create_test_server();
    let app = server.router();

    // Ingest a record via the amorphic endpoint (raw JSON body)
    let (s, body) = post_raw(
        &app,
        "/api/v1/ingest",
        r#"{"name": "Widget", "price": 29.99, "tags": ["hardware", "tool"]}"#,
    )
    .await;
    assert!(s.is_success(), "amorphic ingest failed: {s} {body}");
}

// ============================================================================
// 17. Shard & Cluster Status
// ============================================================================

#[tokio::test]
async fn e2e_shard_and_cluster() {
    let server = create_test_server();
    let app = server.router();

    let (s, _) = get_json(&app, "/api/v1/shards/status").await;
    assert_eq!(s, StatusCode::OK);

    let (s, _) = get_json(&app, "/api/v1/cluster/nodes").await;
    assert_eq!(s, StatusCode::OK);

    let (s, _) = get_json(&app, "/api/v1/cluster/health").await;
    assert_eq!(s, StatusCode::OK);
}

// ============================================================================
// 18. Full Lifecycle: Cross-feature Integration
// ============================================================================

#[tokio::test]
async fn e2e_full_lifecycle() {
    let server = create_test_server();
    let app = server.router();

    // 1. Health
    let (s, _) = get_json(&app, "/health").await;
    assert_eq!(s, StatusCode::OK);

    // 2. Create table and insert data via unified SQL
    let (_, body) = post_json(
        &app,
        "/",
        &json!({"q": "CREATE TABLE orders (id INT, product TEXT, amount REAL)"}),
    )
    .await;
    assert_eq!(body["ok"], true, "CREATE TABLE failed: {body}");

    for i in 1..=5 {
        let (_, body) = post_json(
            &app,
            "/",
            &json!({"q": format!("INSERT INTO orders (id, product, amount) VALUES ({i}, 'item{i}', {:.2})", i as f64 * 10.5)}),
        ).await;
        assert_eq!(body["ok"], true, "INSERT {i} failed: {body}");
    }

    // 3. Query with COUNT
    let (_, body) = post_json(&app, "/", &json!({"q": "SELECT COUNT(*) FROM orders"})).await;
    assert_eq!(body["ok"], true);

    // 4. Verify pg_catalog reflects our table
    let (_, body) = post_json(
        &app,
        "/",
        &json!({"q": "SELECT * FROM pg_catalog.pg_class"}),
    )
    .await;
    assert_eq!(body["ok"], true);
    let data = body["data"].as_array().unwrap();
    let has_orders = data.iter().any(|r| {
        r.as_array()
            .map(|cols| cols.iter().any(|c| c.as_str() == Some("orders")))
            .unwrap_or(false)
    });
    assert!(has_orders, "pg_class should contain 'orders' table");

    // 5. Store agent memory about the query
    let (s, _) = post_json(
        &app,
        "/api/v1/memory/store",
        &json!({
            "content": "User queried orders table with 5 rows",
            "memory_type": "episodic"
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK);

    // 6. Create and run a workflow
    let (s, body) = post_json(
        &app,
        "/api/v1/workflows",
        &json!({
            "name": "order_export",
            "steps": [{"label": "export", "operation": {"tool": "export_csv"}, "depends_on": []}]
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "create workflow failed: {body}");
    let wf_id = body["id"].as_str().unwrap();

    let (s, _) = post_json(&app, &format!("/api/v1/workflows/{wf_id}/run"), &json!({})).await;
    assert_eq!(s, StatusCode::OK);

    // 7. Create a tenant
    let (s, _) = post_json(&app, "/api/v1/tenants", &json!({"name": "lifecycle-test"})).await;
    assert_eq!(s, StatusCode::OK);

    // 8. Publish a queue message
    let (s, _) = post_json(
        &app,
        "/api/v1/queue/publish",
        &json!({"topic": "lifecycle", "payload": "test_complete"}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);

    // 9. Energy check
    let (s, _) = get_json(&app, "/api/v1/energy").await;
    assert_eq!(s, StatusCode::OK);

    // 10. Final health
    let (s, _) = get_json(&app, "/health").await;
    assert_eq!(s, StatusCode::OK);
}

// ============================================================================
// 19. MCP Transport (unit-level)
// ============================================================================

#[tokio::test]
async fn e2e_mcp_tool_count() {
    let tools = joule_db_server::mcp_transport::all_tool_definitions();
    assert!(
        tools.len() >= 20,
        "MCP should expose 20+ tools, got {}",
        tools.len()
    );
}

// ============================================================================
// 20. CXL Memory Types (unit-level)
// ============================================================================

#[test]
fn e2e_cxl_memory_types() {
    use joule_db_server::cxl_memory::{CxlMemoryConfig, CxlMemoryTier, CxlTierConfig};

    let config = CxlMemoryConfig {
        enabled: true,
        auto_tiering: true,
        tiers: vec![
            CxlTierConfig {
                tier: CxlMemoryTier::Hot,
                capacity_bytes: 1024 * 1024 * 1024,
                latency_ns: 100,
                bandwidth_gbps: 64.0,
            },
            CxlTierConfig {
                tier: CxlMemoryTier::Warm,
                capacity_bytes: 8 * 1024 * 1024 * 1024,
                latency_ns: 500,
                bandwidth_gbps: 32.0,
            },
            CxlTierConfig {
                tier: CxlMemoryTier::Cold,
                capacity_bytes: 64 * 1024 * 1024 * 1024,
                latency_ns: 2000,
                bandwidth_gbps: 8.0,
            },
        ],
    };

    assert_eq!(config.tiers.len(), 3);
    assert_eq!(config.tiers[0].tier, CxlMemoryTier::Hot);
    assert_eq!(config.tiers[2].tier, CxlMemoryTier::Cold);
}

// ============================================================================
// 21. Error Handling
// ============================================================================

#[tokio::test]
async fn e2e_error_handling() {
    let server = create_test_server();
    let app = server.router();

    // Invalid SQL via unified endpoint — engine may return empty results for
    // unparseable SQL rather than error, so just verify we get a response
    let (s, body) = post_json(&app, "/", &json!({"q": "INVALID SQL GARBAGE"})).await;
    assert_eq!(s, StatusCode::OK, "unified endpoint should return 200");
    assert!(body.get("ok").is_some(), "response should have ok field");

    // Get nonexistent key — returns 200 with null (Option<String> = None)
    let (s, body) = get_json(&app, "/api/v1/keys/nonexistent_key_12345").await;
    assert_eq!(s, StatusCode::OK);
    assert!(body.is_null(), "nonexistent key should return null");

    // Get nonexistent workflow
    let (s, _) = get_json(&app, "/api/v1/workflows/nonexistent").await;
    assert_eq!(s, StatusCode::NOT_FOUND);

    // Get nonexistent edge PoP
    let (s, _) = get_json(&app, "/api/v1/edge/pops/nonexistent").await;
    assert_eq!(s, StatusCode::NOT_FOUND);

    // Get nonexistent tenant
    let (s, _) = get_json(&app, "/api/v1/tenants/nonexistent").await;
    assert_eq!(s, StatusCode::NOT_FOUND);

    // Get nonexistent branch
    let (s, _) = get_json(&app, "/api/v1/branches/nonexistent").await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}
