//! Integration tests for the unified amorphic storage backend.
//!
//! Tests that SQL queries, amorphic REST endpoints, and KV operations
//! all work correctly with the DurableAmorphicStore as the primary
//! storage engine.

use joule_db_server::amorphic_adapter::AmorphicTableStorage;
use joule_db_server::query::{QueryExecutor, QueryRequest, SimpleQueryExecutor};
use std::sync::Arc;

fn create_executor(dir: &tempfile::TempDir) -> SimpleQueryExecutor {
    let amorphic_path = dir.path().join("amorphic");
    let store = joule_db_amorphic::DurableAmorphicStore::open(amorphic_path.to_str().unwrap())
        .expect("Failed to open amorphic store");
    let storage = Arc::new(AmorphicTableStorage::new(store));
    SimpleQueryExecutor::with_amorphic(storage)
}

fn exec(executor: &SimpleQueryExecutor, sql: &str) -> joule_db_server::query::QueryResponse {
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
    executor
        .execute(&request)
        .expect(&format!("SQL failed: {}", sql))
}

fn exec_err(
    executor: &SimpleQueryExecutor,
    sql: &str,
) -> joule_db_server::query::QueryErrorResponse {
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
    executor
        .execute(&request)
        .expect_err(&format!("Expected error for: {}", sql))
}

// ─── SQL round-trip tests ───────────────────────────────────────────────────

#[test]
fn test_sql_create_insert_select() {
    let dir = tempfile::tempdir().unwrap();
    let executor = create_executor(&dir);

    exec(&executor, "CREATE TABLE users (id INT, name TEXT, age INT)");
    exec(
        &executor,
        "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)",
    );
    exec(
        &executor,
        "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)",
    );

    let resp = exec(&executor, "SELECT id, name, age FROM users");
    assert_eq!(resp.columns, vec!["id", "name", "age"]);
    assert_eq!(resp.rows.len(), 2);
    let mut names: Vec<String> = resp
        .rows
        .iter()
        .map(|r| r[1].as_str().unwrap().to_string())
        .collect();
    names.sort();
    assert_eq!(names, vec!["Alice", "Bob"]);
}

#[test]
fn test_sql_update() {
    let dir = tempfile::tempdir().unwrap();
    let executor = create_executor(&dir);

    exec(
        &executor,
        "CREATE TABLE users (id INT, name TEXT, score INT)",
    );
    exec(
        &executor,
        "INSERT INTO users (id, name, score) VALUES (1, 'Alice', 100)",
    );
    exec(&executor, "UPDATE users SET score = 200 WHERE id = 1");

    let resp = exec(&executor, "SELECT score FROM users WHERE id = 1");
    assert_eq!(resp.rows.len(), 1);
    assert_eq!(resp.rows[0][0], serde_json::json!(200));
}

#[test]
fn test_sql_delete() {
    let dir = tempfile::tempdir().unwrap();
    let executor = create_executor(&dir);

    exec(&executor, "CREATE TABLE items (id INT, name TEXT)");
    exec(
        &executor,
        "INSERT INTO items (id, name) VALUES (1, 'Sword')",
    );
    exec(
        &executor,
        "INSERT INTO items (id, name) VALUES (2, 'Shield')",
    );
    exec(&executor, "DELETE FROM items WHERE id = 1");

    let resp = exec(&executor, "SELECT name FROM items");
    assert_eq!(resp.rows.len(), 1);
    assert_eq!(resp.rows[0][0], serde_json::json!("Shield"));
}

#[test]
fn test_sql_drop_table() {
    let dir = tempfile::tempdir().unwrap();
    let executor = create_executor(&dir);

    exec(&executor, "CREATE TABLE temp (x INT)");
    exec(&executor, "INSERT INTO temp (x) VALUES (1)");
    exec(&executor, "DROP TABLE temp");

    // Querying dropped table should fail
    exec_err(&executor, "SELECT * FROM temp");
}

#[test]
fn test_sql_where_clause() {
    let dir = tempfile::tempdir().unwrap();
    let executor = create_executor(&dir);

    exec(
        &executor,
        "CREATE TABLE products (id INT, name TEXT, price INT)",
    );
    exec(
        &executor,
        "INSERT INTO products (id, name, price) VALUES (1, 'Apple', 1)",
    );
    exec(
        &executor,
        "INSERT INTO products (id, name, price) VALUES (2, 'Banana', 2)",
    );
    exec(
        &executor,
        "INSERT INTO products (id, name, price) VALUES (3, 'Cherry', 5)",
    );

    let resp = exec(&executor, "SELECT name FROM products WHERE price > 1");
    assert_eq!(resp.rows.len(), 2);
    let mut names: Vec<String> = resp
        .rows
        .iter()
        .map(|r| r[0].as_str().unwrap().to_string())
        .collect();
    names.sort();
    assert_eq!(names, vec!["Banana", "Cherry"]);
}

#[test]
fn test_sql_count_aggregate() {
    let dir = tempfile::tempdir().unwrap();
    let executor = create_executor(&dir);

    exec(&executor, "CREATE TABLE events (id INT, type TEXT)");
    exec(
        &executor,
        "INSERT INTO events (id, type) VALUES (1, 'click')",
    );
    exec(
        &executor,
        "INSERT INTO events (id, type) VALUES (2, 'click')",
    );
    exec(
        &executor,
        "INSERT INTO events (id, type) VALUES (3, 'view')",
    );

    let resp = exec(&executor, "SELECT COUNT(*) FROM events");
    assert_eq!(resp.rows.len(), 1);
    assert_eq!(resp.rows[0][0], serde_json::json!(3));
}

// ─── Persistence tests ────────────────────────────────────────────────────

#[test]
fn test_sql_persistence_across_restart() {
    let dir = tempfile::tempdir().unwrap();

    // First "session": create table and insert data
    {
        let executor = create_executor(&dir);
        exec(&executor, "CREATE TABLE persistent (id INT, value TEXT)");
        exec(
            &executor,
            "INSERT INTO persistent (id, value) VALUES (1, 'hello')",
        );
        exec(
            &executor,
            "INSERT INTO persistent (id, value) VALUES (2, 'world')",
        );
    }

    // Second "session": data should survive
    {
        let executor = create_executor(&dir);
        let resp = exec(&executor, "SELECT id, value FROM persistent");
        assert_eq!(resp.rows.len(), 2);
        let mut values: Vec<String> = resp
            .rows
            .iter()
            .map(|r| r[1].as_str().unwrap().to_string())
            .collect();
        values.sort();
        assert_eq!(values, vec!["hello", "world"]);
    }
}

#[test]
fn test_schema_persistence_across_restart() {
    let dir = tempfile::tempdir().unwrap();

    {
        let executor = create_executor(&dir);
        exec(
            &executor,
            "CREATE TABLE typed (id INT, name TEXT, score INT)",
        );
        exec(
            &executor,
            "INSERT INTO typed (id, name, score) VALUES (1, 'test', 42)",
        );
    }

    {
        let executor = create_executor(&dir);
        // Schema should be recovered — columns available
        let resp = exec(&executor, "SELECT name, score FROM typed WHERE id = 1");
        assert_eq!(resp.rows.len(), 1);
        assert_eq!(resp.rows[0][0], serde_json::json!("test"));
        assert_eq!(resp.rows[0][1], serde_json::json!(42));
    }
}

#[test]
fn test_delete_persistence_across_restart() {
    let dir = tempfile::tempdir().unwrap();

    {
        let executor = create_executor(&dir);
        exec(&executor, "CREATE TABLE deletable (id INT, name TEXT)");
        exec(
            &executor,
            "INSERT INTO deletable (id, name) VALUES (1, 'keep')",
        );
        exec(
            &executor,
            "INSERT INTO deletable (id, name) VALUES (2, 'remove')",
        );
        exec(&executor, "DELETE FROM deletable WHERE id = 2");
    }

    {
        let executor = create_executor(&dir);
        let resp = exec(&executor, "SELECT name FROM deletable");
        assert_eq!(resp.rows.len(), 1);
        assert_eq!(resp.rows[0][0], serde_json::json!("keep"));
    }
}

// ─── Amorphic REST adapter tests ──────────────────────────────────────────

#[test]
fn test_amorphic_ingest_and_retrieve() {
    let dir = tempfile::tempdir().unwrap();
    let amorphic_path = dir.path().join("amorphic");
    let store =
        joule_db_amorphic::DurableAmorphicStore::open(amorphic_path.to_str().unwrap()).unwrap();
    let storage = AmorphicTableStorage::new(store);

    let json_str = serde_json::json!({
        "name": "Alice",
        "age": 30,
        "city": "NYC"
    })
    .to_string();
    let id = storage.ingest_json(&json_str).unwrap();

    let record = storage.get_record(id);
    assert!(record.is_some());
    let record = record.unwrap();
    assert_eq!(record["name"], serde_json::json!("Alice"));
}

#[test]
fn test_amorphic_delete() {
    let dir = tempfile::tempdir().unwrap();
    let amorphic_path = dir.path().join("amorphic");
    let store =
        joule_db_amorphic::DurableAmorphicStore::open(amorphic_path.to_str().unwrap()).unwrap();
    let storage = AmorphicTableStorage::new(store);

    let id = storage.ingest_json(r#"{"key": "value"}"#).unwrap();
    assert!(storage.get_record(id).is_some());

    storage.delete_record(id).unwrap();
    assert!(storage.get_record(id).is_none());
}

#[test]
fn test_amorphic_ingest_edge() {
    let dir = tempfile::tempdir().unwrap();
    let amorphic_path = dir.path().join("amorphic");
    let store =
        joule_db_amorphic::DurableAmorphicStore::open(amorphic_path.to_str().unwrap()).unwrap();
    let storage = AmorphicTableStorage::new(store);

    let id = storage.ingest_edge("Alice", "KNOWS", "Bob").unwrap();
    let record = storage.get_record(id);
    assert!(record.is_some());
}

// ─── Concurrent query tests ──────────────────────────────────────────────

#[test]
fn test_concurrent_sql_queries() {
    let dir = tempfile::tempdir().unwrap();
    let executor = Arc::new(create_executor(&dir));

    exec(&executor, "CREATE TABLE concurrent (id INT, thread TEXT)");

    let mut handles = vec![];
    for i in 0..4 {
        let exec = executor.clone();
        handles.push(std::thread::spawn(move || {
            let sql = format!(
                "INSERT INTO concurrent (id, thread) VALUES ({}, 'thread_{}')",
                i, i
            );
            let request = QueryRequest {
                sql,
                params: Default::default(),
                args: vec![],
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            };
            exec.execute(&request).unwrap();
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let resp = exec(&executor, "SELECT COUNT(*) FROM concurrent");
    assert_eq!(resp.rows[0][0], serde_json::json!(4));
}

// ─── Energy response field tests ──────────────────────────────────────────

#[test]
fn test_query_response_has_energy_fields() {
    let dir = tempfile::tempdir().unwrap();
    let executor = create_executor(&dir);

    exec(&executor, "CREATE TABLE energy_test (id INT)");
    let resp = exec(&executor, "SELECT * FROM energy_test");

    // Without energy feature, fields should be None
    assert!(resp.energy_joules.is_none());
    assert!(resp.power_watts.is_none());

    // Verify the fields serialize correctly (skip_serializing_if = None)
    let json = serde_json::to_value(&resp).unwrap();
    assert!(!json.as_object().unwrap().contains_key("energy_joules"));
    assert!(!json.as_object().unwrap().contains_key("power_watts"));
}

// ─── Server creation test ─────────────────────────────────────────────────

#[test]
fn test_server_creates_with_amorphic() {
    let dir = tempfile::tempdir().unwrap();
    let config = joule_db_server::ServerConfig {
        http_addr: "127.0.0.1:0".to_string(),
        tcp_addr: "127.0.0.1:0".to_string(),
        db_path: dir.path().to_str().unwrap().to_string(),
        ..Default::default()
    };

    let server = joule_db_server::Server::new(config);
    assert!(
        server.is_ok(),
        "Server should create successfully with amorphic"
    );
}
