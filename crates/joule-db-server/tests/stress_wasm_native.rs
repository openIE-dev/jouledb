//! Browser/WASM native stress tests.
//!
//! Tests the workloads that browser clients will exercise: CRUD lifecycles,
//! state machine patterns, JSON document operations, full-text search,
//! LLM embedding patterns, bulk operations, and schema evolution.
//!
//! These tests use `SimpleQueryExecutor` directly (same engine as the WASM build)
//! to verify correctness without requiring a browser runtime.

use joule_db_server::query::{QueryExecutor, QueryRequest, QueryResponse, SimpleQueryExecutor};

fn exec(executor: &SimpleQueryExecutor, sql: &str) -> QueryResponse {
    executor
        .execute(&QueryRequest {
            sql: sql.to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: Some(30000),
            branch_id: None,
            tenant_id: None,
        })
        .expect(&format!("SQL failed: {}", sql))
}

fn exec_err(executor: &SimpleQueryExecutor, sql: &str) -> String {
    format!(
        "{:?}",
        executor
            .execute(&QueryRequest {
                sql: sql.to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: Some(30000),
                branch_id: None,
                tenant_id: None,
            })
            .unwrap_err()
    )
}

fn row_count(resp: &QueryResponse) -> usize {
    resp.rows.len()
}

fn first_cell(resp: &QueryResponse) -> String {
    resp.rows[0][0].to_string().trim_matches('"').to_string()
}

// ============================================================================
// CRUD lifecycle
// ============================================================================

#[test]
fn wasm_crud_create_insert_select_drop() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_crud (id INT, name TEXT)");
    exec(&e, "INSERT INTO wasm_crud VALUES (1, 'alice')");
    exec(&e, "INSERT INTO wasm_crud VALUES (2, 'bob')");

    let resp = exec(&e, "SELECT * FROM wasm_crud");
    assert_eq!(row_count(&resp), 2);

    exec(&e, "DROP TABLE wasm_crud");
    let err = exec_err(&e, "SELECT * FROM wasm_crud");
    assert!(
        err.to_lowercase().contains("not found")
            || err.to_lowercase().contains("no such")
            || err.to_lowercase().contains("does not exist"),
        "Expected table not found error, got: {}",
        err
    );
}

#[test]
fn wasm_crud_update() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_upd (id INT, val TEXT)");
    exec(&e, "INSERT INTO wasm_upd VALUES (1, 'old')");
    exec(&e, "UPDATE wasm_upd SET val = 'new' WHERE id = 1");

    let resp = exec(&e, "SELECT val FROM wasm_upd WHERE id = 1");
    assert_eq!(row_count(&resp), 1);
    assert_eq!(first_cell(&resp), "new");
    exec(&e, "DROP TABLE wasm_upd");
}

#[test]
fn wasm_crud_delete() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_del (id INT, val TEXT)");
    exec(&e, "INSERT INTO wasm_del VALUES (1, 'a')");
    exec(&e, "INSERT INTO wasm_del VALUES (2, 'b')");
    exec(&e, "INSERT INTO wasm_del VALUES (3, 'c')");
    exec(&e, "DELETE FROM wasm_del WHERE id = 2");

    let resp = exec(&e, "SELECT * FROM wasm_del");
    assert_eq!(row_count(&resp), 2);
    exec(&e, "DROP TABLE wasm_del");
}

#[test]
fn wasm_crud_select_where_clauses() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_where (id INT, name TEXT, score INT)");
    exec(&e, "INSERT INTO wasm_where VALUES (1, 'alice', 90)");
    exec(&e, "INSERT INTO wasm_where VALUES (2, 'bob', 75)");
    exec(&e, "INSERT INTO wasm_where VALUES (3, 'charlie', 85)");
    exec(&e, "INSERT INTO wasm_where VALUES (4, 'diana', 95)");

    let resp = exec(&e, "SELECT name FROM wasm_where WHERE score > 80");
    assert_eq!(row_count(&resp), 3); // alice, charlie, diana

    let resp = exec(
        &e,
        "SELECT name FROM wasm_where WHERE score >= 90 AND name != 'alice'",
    );
    assert_eq!(row_count(&resp), 1); // diana

    exec(&e, "DROP TABLE wasm_where");
}

#[test]
fn wasm_crud_count_and_aggregate() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_agg (category TEXT, amount INT)");
    exec(&e, "INSERT INTO wasm_agg VALUES ('a', 10)");
    exec(&e, "INSERT INTO wasm_agg VALUES ('a', 20)");
    exec(&e, "INSERT INTO wasm_agg VALUES ('b', 30)");

    let resp = exec(&e, "SELECT COUNT(*) FROM wasm_agg");
    assert_eq!(first_cell(&resp), "3");

    let resp = exec(&e, "SELECT SUM(amount) FROM wasm_agg");
    assert!(
        first_cell(&resp).starts_with("60"),
        "SUM should be 60, got {}",
        first_cell(&resp)
    );

    exec(&e, "DROP TABLE wasm_agg");
}

#[test]
fn wasm_crud_recreate_after_drop() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_recreate (id INT)");
    exec(&e, "INSERT INTO wasm_recreate VALUES (1)");
    exec(&e, "DROP TABLE wasm_recreate");

    // Re-create with different schema
    exec(
        &e,
        "CREATE TABLE wasm_recreate (id INT, name TEXT, active INT)",
    );
    exec(&e, "INSERT INTO wasm_recreate VALUES (1, 'test', 1)");
    let resp = exec(&e, "SELECT * FROM wasm_recreate");
    assert_eq!(row_count(&resp), 1);
    assert_eq!(resp.columns.len(), 3);
    exec(&e, "DROP TABLE wasm_recreate");
}

#[test]
fn wasm_crud_insert_null_values() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_null (id INT, name TEXT)");
    exec(&e, "INSERT INTO wasm_null VALUES (1, NULL)");
    exec(&e, "INSERT INTO wasm_null VALUES (NULL, 'test')");

    let resp = exec(&e, "SELECT * FROM wasm_null");
    assert_eq!(row_count(&resp), 2);
    exec(&e, "DROP TABLE wasm_null");
}

#[test]
fn wasm_crud_order_by() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_order (id INT, name TEXT)");
    exec(&e, "INSERT INTO wasm_order VALUES (3, 'charlie')");
    exec(&e, "INSERT INTO wasm_order VALUES (1, 'alice')");
    exec(&e, "INSERT INTO wasm_order VALUES (2, 'bob')");

    let resp = exec(&e, "SELECT name FROM wasm_order ORDER BY id ASC");
    assert_eq!(row_count(&resp), 3);
    assert_eq!(first_cell(&resp), "alice");
    exec(&e, "DROP TABLE wasm_order");
}

#[test]
fn wasm_crud_limit_offset() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_page (id INT)");
    for i in 1..=20 {
        exec(&e, &format!("INSERT INTO wasm_page VALUES ({})", i));
    }

    let resp = exec(&e, "SELECT * FROM wasm_page LIMIT 5");
    assert_eq!(row_count(&resp), 5);

    let resp = exec(&e, "SELECT * FROM wasm_page LIMIT 5 OFFSET 10");
    assert_eq!(row_count(&resp), 5);

    exec(&e, "DROP TABLE wasm_page");
}

// ============================================================================
// State machine patterns
// ============================================================================

#[test]
fn wasm_state_machine_order_status() {
    let e = SimpleQueryExecutor::new();
    exec(
        &e,
        "CREATE TABLE wasm_orders (id INT, status TEXT, updated_at TEXT)",
    );

    // Create order
    exec(
        &e,
        "INSERT INTO wasm_orders VALUES (1, 'pending', '2024-01-01')",
    );

    // Transition: pending -> processing
    exec(
        &e,
        "UPDATE wasm_orders SET status = 'processing', updated_at = '2024-01-02' WHERE id = 1",
    );
    let resp = exec(&e, "SELECT status FROM wasm_orders WHERE id = 1");
    assert_eq!(first_cell(&resp), "processing");

    // Transition: processing -> shipped
    exec(
        &e,
        "UPDATE wasm_orders SET status = 'shipped', updated_at = '2024-01-03' WHERE id = 1",
    );
    let resp = exec(&e, "SELECT status FROM wasm_orders WHERE id = 1");
    assert_eq!(first_cell(&resp), "shipped");

    // Transition: shipped -> delivered
    exec(
        &e,
        "UPDATE wasm_orders SET status = 'delivered', updated_at = '2024-01-04' WHERE id = 1",
    );
    let resp = exec(&e, "SELECT status FROM wasm_orders WHERE id = 1");
    assert_eq!(first_cell(&resp), "delivered");

    exec(&e, "DROP TABLE wasm_orders");
}

#[test]
fn wasm_state_machine_inventory() {
    let e = SimpleQueryExecutor::new();
    exec(
        &e,
        "CREATE TABLE wasm_inventory (product_id INT, quantity INT, reserved INT)",
    );
    exec(&e, "INSERT INTO wasm_inventory VALUES (100, 50, 0)");

    // Reserve 10 units
    exec(
        &e,
        "UPDATE wasm_inventory SET reserved = reserved + 10 WHERE product_id = 100",
    );
    let resp = exec(
        &e,
        "SELECT reserved FROM wasm_inventory WHERE product_id = 100",
    );
    assert_eq!(first_cell(&resp), "10");

    // Fulfill reservation (decrease quantity and reserved)
    exec(
        &e,
        "UPDATE wasm_inventory SET quantity = quantity - 10, reserved = reserved - 10 WHERE product_id = 100",
    );
    let resp = exec(
        &e,
        "SELECT quantity, reserved FROM wasm_inventory WHERE product_id = 100",
    );
    assert_eq!(row_count(&resp), 1);

    exec(&e, "DROP TABLE wasm_inventory");
}

#[test]
fn wasm_state_machine_session_management() {
    let e = SimpleQueryExecutor::new();
    exec(
        &e,
        "CREATE TABLE wasm_sessions (session_id TEXT, user_id INT, active INT, created TEXT)",
    );

    // Create session
    exec(
        &e,
        "INSERT INTO wasm_sessions VALUES ('sess_abc', 1, 1, '2024-01-01')",
    );

    // Verify session is active
    let resp = exec(
        &e,
        "SELECT user_id FROM wasm_sessions WHERE session_id = 'sess_abc' AND active = 1",
    );
    assert_eq!(row_count(&resp), 1);

    // Deactivate session
    exec(
        &e,
        "UPDATE wasm_sessions SET active = 0 WHERE session_id = 'sess_abc'",
    );
    let resp = exec(
        &e,
        "SELECT user_id FROM wasm_sessions WHERE session_id = 'sess_abc' AND active = 1",
    );
    assert_eq!(row_count(&resp), 0);

    exec(&e, "DROP TABLE wasm_sessions");
}

#[test]
fn wasm_state_machine_counter() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_counters (name TEXT, value INT)");
    exec(&e, "INSERT INTO wasm_counters VALUES ('page_views', 0)");

    // Increment 100 times
    for _ in 0..100 {
        exec(
            &e,
            "UPDATE wasm_counters SET value = value + 1 WHERE name = 'page_views'",
        );
    }

    let resp = exec(
        &e,
        "SELECT value FROM wasm_counters WHERE name = 'page_views'",
    );
    assert_eq!(first_cell(&resp), "100");

    exec(&e, "DROP TABLE wasm_counters");
}

// ============================================================================
// JSON document store patterns
// ============================================================================

#[test]
fn wasm_json_store_and_extract() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_docs (id INT, doc TEXT)");
    exec(
        &e,
        r#"INSERT INTO wasm_docs VALUES (1, '{"name":"alice","age":30,"tags":["admin","user"]}')"#,
    );
    exec(
        &e,
        r#"INSERT INTO wasm_docs VALUES (2, '{"name":"bob","age":25,"tags":["user"]}')"#,
    );

    let resp = exec(
        &e,
        "SELECT JSON_EXTRACT(doc, '$.name') FROM wasm_docs WHERE id = 1",
    );
    assert_eq!(row_count(&resp), 1);

    let resp = exec(
        &e,
        "SELECT JSON_EXTRACT(doc, '$.age') FROM wasm_docs WHERE id = 2",
    );
    assert_eq!(row_count(&resp), 1);

    exec(&e, "DROP TABLE wasm_docs");
}

#[test]
fn wasm_json_nested_extraction() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_nested (id INT, data TEXT)");
    exec(
        &e,
        r#"INSERT INTO wasm_nested VALUES (1, '{"user":{"name":"alice","address":{"city":"NYC"}}}')"#,
    );

    let resp = exec(
        &e,
        "SELECT JSON_EXTRACT(data, '$.user.name') FROM wasm_nested WHERE id = 1",
    );
    assert_eq!(row_count(&resp), 1);

    exec(&e, "DROP TABLE wasm_nested");
}

#[test]
fn wasm_json_type_checking() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_jtype (id INT, val TEXT)");
    exec(
        &e,
        r#"INSERT INTO wasm_jtype VALUES (1, '{"a":1,"b":"str","c":true,"d":null,"e":[1,2]}')"#,
    );

    let resp = exec(&e, "SELECT JSON_TYPE(val) FROM wasm_jtype");
    assert_eq!(row_count(&resp), 1);

    exec(&e, "DROP TABLE wasm_jtype");
}

#[test]
fn wasm_json_array_length() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_jarr (id INT, arr TEXT)");
    exec(&e, r#"INSERT INTO wasm_jarr VALUES (1, '[1,2,3,4,5]')"#);

    let resp = exec(&e, "SELECT JSON_ARRAY_LENGTH(arr) FROM wasm_jarr");
    assert_eq!(row_count(&resp), 1);
    assert_eq!(first_cell(&resp), "5");

    exec(&e, "DROP TABLE wasm_jarr");
}

// ============================================================================
// Browser LLM patterns
// ============================================================================

#[test]
fn wasm_llm_conversation_history() {
    let e = SimpleQueryExecutor::new();
    exec(
        &e,
        "CREATE TABLE wasm_chat (id INT, role TEXT, content TEXT, timestamp INT)",
    );
    exec(
        &e,
        "INSERT INTO wasm_chat VALUES (1, 'user', 'Hello!', 1000)",
    );
    exec(
        &e,
        "INSERT INTO wasm_chat VALUES (2, 'assistant', 'Hi there!', 1001)",
    );
    exec(
        &e,
        "INSERT INTO wasm_chat VALUES (3, 'user', 'How are you?', 1002)",
    );
    exec(
        &e,
        "INSERT INTO wasm_chat VALUES (4, 'assistant', 'I am well!', 1003)",
    );

    // Get last N messages (sliding window for context)
    let resp = exec(
        &e,
        "SELECT role, content FROM wasm_chat ORDER BY timestamp DESC LIMIT 3",
    );
    assert_eq!(row_count(&resp), 3);

    // Count messages by role
    let resp = exec(&e, "SELECT role, COUNT(*) FROM wasm_chat GROUP BY role");
    assert_eq!(row_count(&resp), 2);

    exec(&e, "DROP TABLE wasm_chat");
}

#[test]
fn wasm_llm_prompt_template_store() {
    let e = SimpleQueryExecutor::new();
    exec(
        &e,
        "CREATE TABLE wasm_prompts (id INT, name TEXT, template TEXT, version INT)",
    );
    exec(
        &e,
        "INSERT INTO wasm_prompts VALUES (1, 'summarize', 'Summarize: {{text}}', 1)",
    );
    exec(
        &e,
        "INSERT INTO wasm_prompts VALUES (2, 'translate', 'Translate to {{lang}}: {{text}}', 1)",
    );

    // Version update
    exec(
        &e,
        "INSERT INTO wasm_prompts VALUES (3, 'summarize', 'Concisely summarize: {{text}}', 2)",
    );

    // Get latest version of a prompt
    let resp = exec(
        &e,
        "SELECT template FROM wasm_prompts WHERE name = 'summarize' ORDER BY version DESC LIMIT 1",
    );
    assert_eq!(row_count(&resp), 1);
    assert!(first_cell(&resp).contains("Concisely"));

    exec(&e, "DROP TABLE wasm_prompts");
}

#[test]
fn wasm_llm_token_counting() {
    let e = SimpleQueryExecutor::new();
    exec(
        &e,
        "CREATE TABLE wasm_tokens (id INT, text TEXT, token_count INT)",
    );
    exec(&e, "INSERT INTO wasm_tokens VALUES (1, 'Hello world', 2)");
    exec(
        &e,
        "INSERT INTO wasm_tokens VALUES (2, 'This is a longer sentence with more tokens', 8)",
    );

    let resp = exec(&e, "SELECT SUM(token_count) FROM wasm_tokens");
    assert!(
        first_cell(&resp).starts_with("10"),
        "SUM should be 10, got {}",
        first_cell(&resp)
    );

    // Check if under context window limit
    let resp = exec(&e, "SELECT SUM(token_count) < 4096 FROM wasm_tokens");
    assert!(row_count(&resp) >= 1, "Should return at least 1 row");

    exec(&e, "DROP TABLE wasm_tokens");
}

#[test]
fn wasm_llm_embedding_storage() {
    let e = SimpleQueryExecutor::new();
    exec(
        &e,
        "CREATE TABLE wasm_embeddings (id INT, text TEXT, dim0 REAL, dim1 REAL, dim2 REAL, dim3 REAL)",
    );
    exec(
        &e,
        "INSERT INTO wasm_embeddings VALUES (1, 'cat', 0.1, 0.9, 0.2, 0.3)",
    );
    exec(
        &e,
        "INSERT INTO wasm_embeddings VALUES (2, 'dog', 0.15, 0.85, 0.25, 0.35)",
    );
    exec(
        &e,
        "INSERT INTO wasm_embeddings VALUES (3, 'car', 0.9, 0.1, 0.8, 0.7)",
    );

    // Cosine similarity approximation: find similar to 'cat'
    let resp = exec(
        &e,
        "SELECT text FROM wasm_embeddings WHERE dim0 < 0.5 AND dim1 > 0.5",
    );
    assert_eq!(row_count(&resp), 2); // cat and dog

    exec(&e, "DROP TABLE wasm_embeddings");
}

// ============================================================================
// Bulk operations
// ============================================================================

#[test]
fn wasm_bulk_insert_1000_rows() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_bulk (id INT, val TEXT)");

    for i in 0..1000 {
        exec(
            &e,
            &format!("INSERT INTO wasm_bulk VALUES ({}, 'row_{}')", i, i),
        );
    }

    let resp = exec(&e, "SELECT COUNT(*) FROM wasm_bulk");
    assert_eq!(first_cell(&resp), "1000");

    exec(&e, "DROP TABLE wasm_bulk");
}

#[test]
fn wasm_bulk_paginated_select() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_pages (id INT)");
    for i in 1..=100 {
        exec(&e, &format!("INSERT INTO wasm_pages VALUES ({})", i));
    }

    // Paginate through all rows, 10 at a time
    let mut total = 0;
    for page in 0..10 {
        let resp = exec(
            &e,
            &format!("SELECT * FROM wasm_pages LIMIT 10 OFFSET {}", page * 10),
        );
        total += row_count(&resp);
    }
    assert_eq!(total, 100);

    exec(&e, "DROP TABLE wasm_pages");
}

#[test]
fn wasm_bulk_delete_with_condition() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_bdel (id INT, active INT)");
    for i in 1..=50 {
        let active = if i % 2 == 0 { 1 } else { 0 };
        exec(
            &e,
            &format!("INSERT INTO wasm_bdel VALUES ({}, {})", i, active),
        );
    }

    exec(&e, "DELETE FROM wasm_bdel WHERE active = 0");
    let resp = exec(&e, "SELECT COUNT(*) FROM wasm_bdel");
    assert_eq!(first_cell(&resp), "25");

    exec(&e, "DROP TABLE wasm_bdel");
}

// ============================================================================
// Schema evolution
// ============================================================================

#[test]
fn wasm_schema_alter_add_column() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_evolve (id INT, name TEXT)");
    exec(&e, "INSERT INTO wasm_evolve VALUES (1, 'alice')");

    exec(&e, "ALTER TABLE wasm_evolve ADD COLUMN email TEXT");
    exec(
        &e,
        "INSERT INTO wasm_evolve VALUES (2, 'bob', 'bob@example.com')",
    );

    let resp = exec(&e, "SELECT * FROM wasm_evolve");
    assert_eq!(row_count(&resp), 2);
    assert!(
        resp.columns.len() >= 3,
        "Should have at least 3 columns after ALTER"
    );

    exec(&e, "DROP TABLE wasm_evolve");
}

#[test]
fn wasm_schema_query_after_alter() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_alter2 (id INT)");
    exec(&e, "INSERT INTO wasm_alter2 VALUES (1)");

    exec(&e, "ALTER TABLE wasm_alter2 ADD COLUMN name TEXT");

    // Query with new column — old rows should have NULL for new column
    let resp = exec(&e, "SELECT id, name FROM wasm_alter2");
    assert_eq!(row_count(&resp), 1);

    exec(&e, "DROP TABLE wasm_alter2");
}

// ============================================================================
// Concurrent browser tabs (simulated with threads)
// ============================================================================

#[test]
fn wasm_concurrent_reads() {
    use std::sync::Arc;
    use std::thread;

    let e = Arc::new(SimpleQueryExecutor::new());
    exec(&e, "CREATE TABLE wasm_conc_r (id INT, val TEXT)");
    for i in 0..100 {
        exec(
            &e,
            &format!("INSERT INTO wasm_conc_r VALUES ({}, 'val_{}')", i, i),
        );
    }

    let handles: Vec<_> = (0..10)
        .map(|_| {
            let e = e.clone();
            thread::spawn(move || {
                let resp = QueryExecutor::execute(
                    e.as_ref(),
                    &QueryRequest {
                        sql: "SELECT COUNT(*) FROM wasm_conc_r".to_string(),
                        params: Default::default(),
                        args: Vec::new(),
                        explain: false,
                        limit: None,
                        session_id: None,
                        query_timeout_ms: Some(30000),
                        branch_id: None,
                        tenant_id: None,
                    },
                )
                .unwrap();
                row_count(&resp)
            })
        })
        .collect();

    for h in handles {
        assert_eq!(h.join().unwrap(), 1);
    }

    exec(&e, "DROP TABLE wasm_conc_r");
}

// ============================================================================
// Error handling
// ============================================================================

#[test]
fn wasm_error_invalid_sql() {
    let e = SimpleQueryExecutor::new();
    let err = exec_err(&e, "THIS IS NOT SQL");
    assert!(
        !err.is_empty(),
        "Should return error message for invalid SQL"
    );
}

#[test]
fn wasm_error_insert_type_mismatch() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_type_err (id INT, name TEXT)");
    // This should either succeed (implicit conversion) or return a clear error
    let result = e.execute(&QueryRequest {
        sql: "INSERT INTO wasm_type_err VALUES ('not_a_number', 123)".to_string(),
        params: Default::default(),
        args: Vec::new(),
        explain: false,
        limit: None,
        session_id: None,
        query_timeout_ms: Some(30000),
        branch_id: None,
        tenant_id: None,
    });
    // Either way, no panic
    let _ = result;
    exec(&e, "DROP TABLE wasm_type_err");
}

#[test]
fn wasm_error_duplicate_table() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_dup (id INT)");
    let err = exec_err(&e, "CREATE TABLE wasm_dup (id INT)");
    assert!(!err.is_empty(), "Should error on duplicate table creation");
    exec(&e, "DROP TABLE wasm_dup");
}

// ============================================================================
// String operations (browser content)
// ============================================================================

#[test]
fn wasm_string_operations() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_str (id INT, text TEXT)");
    exec(&e, "INSERT INTO wasm_str VALUES (1, 'Hello World')");
    exec(&e, "INSERT INTO wasm_str VALUES (2, '  spaces  ')");

    let resp = exec(&e, "SELECT UPPER(text) FROM wasm_str WHERE id = 1");
    assert_eq!(first_cell(&resp), "HELLO WORLD");

    let resp = exec(&e, "SELECT LOWER(text) FROM wasm_str WHERE id = 1");
    assert_eq!(first_cell(&resp), "hello world");

    let resp = exec(&e, "SELECT LENGTH(text) FROM wasm_str WHERE id = 1");
    assert_eq!(first_cell(&resp), "11");

    let resp = exec(&e, "SELECT TRIM(text) FROM wasm_str WHERE id = 2");
    assert_eq!(first_cell(&resp), "spaces");

    exec(&e, "DROP TABLE wasm_str");
}

#[test]
fn wasm_emoji_storage() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE wasm_emoji (id INT, content TEXT)");
    exec(
        &e,
        "INSERT INTO wasm_emoji VALUES (1, '\u{1F680}\u{1F30D}\u{2764}')",
    );

    let resp = exec(&e, "SELECT content FROM wasm_emoji WHERE id = 1");
    assert_eq!(first_cell(&resp), "\u{1F680}\u{1F30D}\u{2764}");

    let resp = exec(&e, "SELECT LENGTH(content) FROM wasm_emoji WHERE id = 1");
    assert_eq!(first_cell(&resp), "3"); // character count, not byte count

    exec(&e, "DROP TABLE wasm_emoji");
}
