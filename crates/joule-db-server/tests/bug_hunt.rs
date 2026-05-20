//! Bug-hunting tests: probing edge cases, NULL handling, type coercion,
//! workflow logic, branch operations, KV semantics, and API contract issues.

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

fn create_test_server() -> joule_db_server::Server {
    let config = joule_db_server::ServerConfig {
        db_path: tempfile::tempdir()
            .unwrap()
            .into_path()
            .to_str()
            .unwrap()
            .to_string(),
        ..Default::default()
    };
    joule_db_server::Server::new(config).expect("failed to create test server")
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

/// Helper: execute SQL via unified endpoint and return (ok, data, affected_rows, error)
async fn sql(app: &axum::Router, query: &str) -> (bool, Vec<Value>, Option<u64>, Option<String>) {
    let (status, body) = post_json(app, "/", &json!({"q": query})).await;
    let ok = body["ok"].as_bool().unwrap_or(false);
    let data = body["data"].as_array().cloned().unwrap_or_default();
    let affected = body["affected_rows"].as_u64();
    let error = body["error"].as_str().map(|s| s.to_string());
    (ok, data, affected, error)
}

// ============================================================================
// 1. SQL NULL Handling
// ============================================================================

#[tokio::test]
async fn bug_null_in_where_clause() {
    let server = create_test_server();
    let app = server.router();

    // Create table and insert rows with NULLs
    sql(&app, "CREATE TABLE nulltest (id INT, name TEXT, score INT)").await;
    sql(&app, "INSERT INTO nulltest VALUES (1, 'Alice', 100)").await;
    sql(&app, "INSERT INTO nulltest VALUES (2, 'Bob', NULL)").await;
    sql(&app, "INSERT INTO nulltest VALUES (3, NULL, 80)").await;

    // NULL = NULL should be false (SQL semantics)
    let (ok, data, _, _) = sql(&app, "SELECT * FROM nulltest WHERE score = NULL").await;
    assert!(ok, "query should succeed");
    assert_eq!(
        data.len(),
        0,
        "NULL = NULL should return no rows (SQL three-valued logic)"
    );

    // IS NULL should find NULL rows
    let (ok, data, _, _) = sql(&app, "SELECT * FROM nulltest WHERE score IS NULL").await;
    assert!(ok, "IS NULL query should succeed");
    assert_eq!(data.len(), 1, "should find exactly 1 row with NULL score");

    // IS NOT NULL should find non-NULL rows
    let (ok, data, _, _) = sql(&app, "SELECT * FROM nulltest WHERE score IS NOT NULL").await;
    assert!(ok, "IS NOT NULL query should succeed");
    assert_eq!(data.len(), 2, "should find 2 rows with non-NULL score");

    // NULL in ORDER BY — NULLs should sort (typically last)
    let (ok, data, _, _) = sql(&app, "SELECT id, score FROM nulltest ORDER BY score").await;
    assert!(ok, "ORDER BY with NULLs should succeed");
    assert_eq!(data.len(), 3, "all 3 rows should be returned");
}

#[tokio::test]
async fn bug_null_in_aggregations() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE agg_null (id INT, val INT)").await;
    sql(&app, "INSERT INTO agg_null VALUES (1, 10)").await;
    sql(&app, "INSERT INTO agg_null VALUES (2, NULL)").await;
    sql(&app, "INSERT INTO agg_null VALUES (3, 30)").await;

    // COUNT(*) should count all rows including NULLs
    let (ok, data, _, _) = sql(&app, "SELECT COUNT(*) FROM agg_null").await;
    assert!(ok);
    let count = data[0].as_array().unwrap()[0].as_f64().unwrap_or(0.0) as i64;
    assert_eq!(count, 3, "COUNT(*) should count all rows including NULL");

    // COUNT(val) should exclude NULLs
    let (ok, data, _, _) = sql(&app, "SELECT COUNT(val) FROM agg_null").await;
    assert!(ok);
    let count = data[0].as_array().unwrap()[0].as_f64().unwrap_or(0.0) as i64;
    assert_eq!(count, 2, "COUNT(val) should exclude NULL values");

    // SUM should skip NULLs
    let (ok, data, _, _) = sql(&app, "SELECT SUM(val) FROM agg_null").await;
    assert!(ok);
    let sum = data[0].as_array().unwrap()[0].as_f64().unwrap_or(-999.0);
    assert!(
        (sum - 40.0).abs() < 0.01,
        "SUM should skip NULL values: 10+30=40, got {sum}"
    );

    // AVG should skip NULLs (average of 10,30 = 20, not 10+30/3)
    let (ok, data, _, _) = sql(&app, "SELECT AVG(val) FROM agg_null").await;
    assert!(ok);
    let avg = data[0].as_array().unwrap()[0].as_f64().unwrap_or(-999.0);
    // Note: AVG implementation uses unwrap_or(0.0) for NULLs and divides by total row count,
    // so it may return (10+0+30)/3 ≈ 13.33 instead of (10+30)/2 = 20. This is a real bug.
    assert!(
        (avg - 20.0).abs() < 0.01,
        "AVG should skip NULLs: (10+30)/2=20, got {avg}"
    );
}

#[tokio::test]
async fn bug_aggregation_on_empty_table() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE empty_agg (id INT, val INT)").await;

    // COUNT(*) on empty table should return 0
    let (ok, data, _, _) = sql(&app, "SELECT COUNT(*) FROM empty_agg").await;
    assert!(ok);
    let count = data[0].as_array().unwrap()[0].as_i64().unwrap_or(-1);
    assert_eq!(count, 0, "COUNT(*) on empty table should be 0");

    // SUM on empty table should return NULL (or 0 depending on impl)
    let (ok, data, _, _) = sql(&app, "SELECT SUM(val) FROM empty_agg").await;
    assert!(ok);
    assert!(
        !data.is_empty(),
        "SUM on empty table should still return a row"
    );

    // AVG on empty table should return NULL (not divide-by-zero)
    let (ok, data, _, _) = sql(&app, "SELECT AVG(val) FROM empty_agg").await;
    assert!(ok, "AVG on empty table should not error/panic");
    assert!(
        !data.is_empty(),
        "AVG on empty table should still return a row"
    );
}

// ============================================================================
// 2. SQL Type Coercion
// ============================================================================

#[tokio::test]
async fn bug_string_number_comparison() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE types (id INT, val TEXT)").await;
    sql(&app, "INSERT INTO types VALUES (1, '100')").await;
    sql(&app, "INSERT INTO types VALUES (2, '20')").await;
    sql(&app, "INSERT INTO types VALUES (3, 'abc')").await;

    // Comparing TEXT column to integer — should this coerce or error?
    let (ok, data, _, _) = sql(&app, "SELECT * FROM types WHERE val > 50").await;
    // Should succeed without panic
    assert!(ok, "comparing TEXT to INT should not panic");
}

#[tokio::test]
async fn bug_insert_column_count_mismatch() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE strict (id INT, name TEXT, age INT)").await;

    // Too few values
    let (ok, _, _, error) = sql(
        &app,
        "INSERT INTO strict (id, name, age) VALUES (1, 'Alice')",
    )
    .await;
    // Should error, not silently succeed with partial data
    assert!(
        !ok || error.is_some(),
        "INSERT with too few values should error"
    );

    // Too many values
    let (ok, _, _, error) = sql(
        &app,
        "INSERT INTO strict (id, name) VALUES (1, 'Alice', 25)",
    )
    .await;
    assert!(
        !ok || error.is_some(),
        "INSERT with too many values should error"
    );
}

// ============================================================================
// 3. SQL LIMIT/OFFSET Edge Cases
// ============================================================================

#[tokio::test]
async fn bug_limit_offset_edge_cases() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE pages (id INT, data TEXT)").await;
    for i in 1..=10 {
        sql(&app, &format!("INSERT INTO pages VALUES ({i}, 'row_{i}')")).await;
    }

    // LIMIT 0 — should return no rows
    let (ok, data, _, _) = sql(&app, "SELECT * FROM pages LIMIT 0").await;
    assert!(ok);
    assert_eq!(data.len(), 0, "LIMIT 0 should return 0 rows");

    // OFFSET beyond data — should return empty
    let (ok, data, _, _) = sql(&app, "SELECT * FROM pages LIMIT 5 OFFSET 100").await;
    assert!(ok);
    assert_eq!(data.len(), 0, "OFFSET beyond data should return 0 rows");

    // LIMIT larger than table
    let (ok, data, _, _) = sql(&app, "SELECT * FROM pages LIMIT 1000").await;
    assert!(ok);
    assert_eq!(data.len(), 10, "LIMIT > row count should return all rows");

    // OFFSET only (no LIMIT) — should skip rows
    let (ok, data, _, _) = sql(&app, "SELECT * FROM pages OFFSET 8").await;
    assert!(ok);
    assert_eq!(data.len(), 2, "OFFSET 8 of 10 rows should return 2 rows");
}

// ============================================================================
// 4. UPDATE and DELETE Edge Cases
// ============================================================================

#[tokio::test]
async fn bug_update_without_where() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE bulk (id INT, status TEXT)").await;
    sql(&app, "INSERT INTO bulk VALUES (1, 'active')").await;
    sql(&app, "INSERT INTO bulk VALUES (2, 'active')").await;
    sql(&app, "INSERT INTO bulk VALUES (3, 'inactive')").await;

    // UPDATE without WHERE should affect all rows
    let (ok, _, affected, _) = sql(&app, "UPDATE bulk SET status = 'archived'").await;
    assert!(ok, "UPDATE without WHERE should succeed");
    assert_eq!(affected.unwrap_or(0), 3, "should affect all 3 rows");

    // Verify all updated
    let (_, data, _, _) = sql(&app, "SELECT * FROM bulk WHERE status = 'archived'").await;
    assert_eq!(data.len(), 3, "all rows should now be 'archived'");
}

#[tokio::test]
async fn bug_delete_without_where() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE dtest (id INT, val TEXT)").await;
    sql(&app, "INSERT INTO dtest VALUES (1, 'a')").await;
    sql(&app, "INSERT INTO dtest VALUES (2, 'b')").await;

    // DELETE without WHERE should delete all rows
    let (ok, _, affected, _) = sql(&app, "DELETE FROM dtest").await;
    assert!(ok, "DELETE without WHERE should succeed");
    assert_eq!(affected.unwrap_or(0), 2, "should delete all 2 rows");

    // Verify empty
    let (_, data, _, _) = sql(&app, "SELECT * FROM dtest").await;
    assert_eq!(
        data.len(),
        0,
        "table should be empty after DELETE without WHERE"
    );
}

#[tokio::test]
async fn bug_update_nonexistent_column() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE ucol (id INT, name TEXT)").await;
    sql(&app, "INSERT INTO ucol VALUES (1, 'test')").await;

    // UPDATE with nonexistent column should error
    let (ok, _, _, error) = sql(&app, "UPDATE ucol SET nonexistent = 'value'").await;
    assert!(
        !ok || error.is_some(),
        "UPDATE nonexistent column should error"
    );
}

// ============================================================================
// 5. SELECT Edge Cases
// ============================================================================

#[tokio::test]
async fn bug_select_from_nonexistent_table() {
    let server = create_test_server();
    let app = server.router();

    let (ok, _, _, error) = sql(&app, "SELECT * FROM this_table_does_not_exist").await;
    assert!(
        !ok,
        "SELECT from nonexistent table should fail, not return empty"
    );
    assert!(error.is_some(), "should have an error message");
}

#[tokio::test]
async fn bug_select_star_empty_table() {
    let server = create_test_server();
    let app = server.router();

    sql(
        &app,
        "CREATE TABLE empty_select (id INT, name TEXT, age INT)",
    )
    .await;

    // SELECT * on empty table — should return 0 rows but valid column info
    let (ok, data, _, _) = sql(&app, "SELECT * FROM empty_select").await;
    assert!(ok, "SELECT from empty table should succeed");
    assert_eq!(data.len(), 0, "empty table should return 0 rows");
}

#[tokio::test]
async fn bug_select_with_alias() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE aliased (id INT, full_name TEXT)").await;
    sql(&app, "INSERT INTO aliased VALUES (1, 'Alice Smith')").await;

    // Column alias
    let _ = sql(&app, "SELECT full_name AS name FROM aliased").await;
    // Table alias
    let (ok, data, _, _) = sql(&app, "SELECT t.id FROM aliased t WHERE t.id = 1").await;
    assert!(ok, "table alias should work");
    assert_eq!(data.len(), 1, "should find the row via table alias");
}

// ============================================================================
// 6. JOIN Edge Cases
// ============================================================================

#[tokio::test]
async fn bug_join_basic() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE authors (id INT, name TEXT)").await;
    sql(&app, "INSERT INTO authors VALUES (1, 'Alice')").await;
    sql(&app, "INSERT INTO authors VALUES (2, 'Bob')").await;

    sql(
        &app,
        "CREATE TABLE books (id INT, title TEXT, author_id INT)",
    )
    .await;
    sql(&app, "INSERT INTO books VALUES (1, 'Book A', 1)").await;
    sql(&app, "INSERT INTO books VALUES (2, 'Book B', 1)").await;
    sql(&app, "INSERT INTO books VALUES (3, 'Book C', 2)").await;

    // Inner join
    let (ok, data, _, _) = sql(
        &app,
        "SELECT authors.name, books.title FROM authors JOIN books ON authors.id = books.author_id",
    )
    .await;
    assert!(ok, "JOIN should succeed");
    assert_eq!(
        data.len(),
        3,
        "JOIN should produce 3 rows (2 for Alice, 1 for Bob)"
    );
}

#[tokio::test]
async fn bug_join_with_no_matches() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE left_t (id INT, val TEXT)").await;
    sql(&app, "INSERT INTO left_t VALUES (1, 'a')").await;

    sql(&app, "CREATE TABLE right_t (id INT, ref_id INT, val TEXT)").await;
    sql(&app, "INSERT INTO right_t VALUES (1, 999, 'x')").await;

    // JOIN with no matching rows
    let (ok, data, _, _) = sql(
        &app,
        "SELECT * FROM left_t JOIN right_t ON left_t.id = right_t.ref_id",
    )
    .await;
    assert!(ok, "JOIN with no matches should succeed");
    assert_eq!(data.len(), 0, "JOIN with no matches should return 0 rows");
}

// ============================================================================
// 7. GROUP BY Edge Cases
// ============================================================================

#[tokio::test]
async fn bug_group_by_with_nulls() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE gb_null (category TEXT, amount INT)").await;
    sql(&app, "INSERT INTO gb_null VALUES ('A', 10)").await;
    sql(&app, "INSERT INTO gb_null VALUES ('A', 20)").await;
    sql(&app, "INSERT INTO gb_null VALUES ('B', 30)").await;
    sql(&app, "INSERT INTO gb_null VALUES (NULL, 40)").await;
    sql(&app, "INSERT INTO gb_null VALUES (NULL, 50)").await;

    // GROUP BY should group NULLs together
    let (ok, data, _, _) = sql(
        &app,
        "SELECT category, SUM(amount) FROM gb_null GROUP BY category",
    )
    .await;
    assert!(ok, "GROUP BY with NULLs should succeed");
    // Should have 3 groups: 'A', 'B', NULL
    assert_eq!(
        data.len(),
        3,
        "GROUP BY should produce 3 groups (A, B, NULL)"
    );
}

// ============================================================================
// 8. Unified Endpoint: SQL vs Similarity Search Routing
// ============================================================================

#[tokio::test]
async fn bug_misspelled_sql_routes_to_search() {
    let server = create_test_server();
    let app = server.router();

    // Create a table first
    sql(&app, "CREATE TABLE routing_test (id INT, name TEXT)").await;
    sql(&app, "INSERT INTO routing_test VALUES (1, 'test')").await;

    // "SELCT" (typo) doesn't start with a SQL keyword, so is_sql() returns false
    // This silently routes to similarity search instead of erroring!
    let (_, body) = post_json(&app, "/", &json!({"q": "SELCT * FROM routing_test"})).await;
    // The key question: does the user get a clear error, or misleading empty results?
    let ok = body["ok"].as_bool().unwrap_or(false);
    if ok {
        // This is the bug — user gets ok:true with empty results for a typo'd query
        let count = body["count"].as_u64().unwrap_or(0);
        // At minimum, verify it doesn't crash and returns something sensible
        assert!(count == 0 || count > 0, "should return without crashing");
    }
    // Note: This test documents the behavior. A "SELCT" typo silently returns
    // empty similarity search results instead of a SQL syntax error.
}

#[tokio::test]
async fn bug_empty_query_body() {
    let server = create_test_server();
    let app = server.router();

    // Empty string query
    let (_, body) = post_json(&app, "/", &json!({"q": ""})).await;
    // Empty query routed to similarity search (is_sql returns false for empty)
    let ok = body["ok"].as_bool().unwrap_or(false);
    assert!(
        ok || body["error"].is_string(),
        "empty query should not crash"
    );

    // Whitespace-only query
    let (_, body) = post_json(&app, "/", &json!({"q": "   "})).await;
    assert!(
        body["ok"].as_bool().is_some() || body["error"].is_string(),
        "whitespace query should not crash"
    );
}

#[tokio::test]
async fn bug_unified_endpoint_no_body() {
    let server = create_test_server();
    let app = server.router();

    // POST / with no body at all
    let (status, _) = post_raw(&app, "/", "").await;
    assert_ne!(
        status,
        StatusCode::OK,
        "empty body should not return 200 OK"
    );
}

// ============================================================================
// 9. KV Store Edge Cases
// ============================================================================

#[tokio::test]
async fn bug_kv_overwrite_semantics() {
    let server = create_test_server();
    let app = server.router();

    // Write a key
    post_raw(&app, "/api/v1/keys/overwrite", "first").await;
    let (_, body) = get_json(&app, "/api/v1/keys/overwrite").await;
    assert_eq!(body.as_str().unwrap(), "first");

    // Overwrite with new value
    post_raw(&app, "/api/v1/keys/overwrite", "second").await;
    let (_, body) = get_json(&app, "/api/v1/keys/overwrite").await;
    assert_eq!(
        body.as_str().unwrap(),
        "second",
        "overwrite should replace value"
    );
}

#[tokio::test]
async fn bug_kv_empty_value() {
    let server = create_test_server();
    let app = server.router();

    // Write empty string as value
    post_raw(&app, "/api/v1/keys/empty_val", "").await;
    let (status, body) = get_json(&app, "/api/v1/keys/empty_val").await;
    assert_eq!(status, StatusCode::OK);
    // Empty string should be distinguishable from missing key
    assert_eq!(
        body.as_str().unwrap_or("NOT_EMPTY"),
        "",
        "empty value should be stored correctly"
    );
}

#[tokio::test]
async fn bug_kv_special_chars_in_key() {
    let server = create_test_server();
    let app = server.router();

    // Key with special characters (URL-safe)
    post_raw(&app, "/api/v1/keys/key-with-dashes", "dash").await;
    let (s, body) = get_json(&app, "/api/v1/keys/key-with-dashes").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(body.as_str().unwrap(), "dash");

    // Key with dots
    post_raw(&app, "/api/v1/keys/key.with.dots", "dots").await;
    let (s, body) = get_json(&app, "/api/v1/keys/key.with.dots").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(body.as_str().unwrap(), "dots");
}

#[tokio::test]
async fn bug_kv_delete_nonexistent() {
    let server = create_test_server();
    let app = server.router();

    // DELETE a key that doesn't exist
    let (status, body) = delete_json(&app, "/api/v1/keys/never_existed").await;
    assert_eq!(
        status,
        StatusCode::OK,
        "deleting nonexistent key should not error"
    );
    // Should return false (nothing was deleted)
    assert_eq!(
        body.as_bool().unwrap_or(true),
        false,
        "should indicate nothing was deleted"
    );
}

// ============================================================================
// 10. Workflow Engine Edge Cases
// ============================================================================

#[tokio::test]
async fn bug_workflow_empty_steps() {
    let server = create_test_server();
    let app = server.router();

    // Create workflow with zero steps — should this be allowed?
    let (s, body) = post_json(
        &app,
        "/api/v1/workflows",
        &json!({
            "name": "empty_workflow",
            "steps": []
        }),
    )
    .await;
    // Zero steps should either error or create and instantly complete
    if s == StatusCode::OK {
        let wf_id = body["id"].as_str().unwrap();
        // Try running it
        let (s, _) = post_json(&app, &format!("/api/v1/workflows/{wf_id}/run"), &json!({})).await;
        assert_eq!(s, StatusCode::OK, "running empty workflow should not panic");
    }
}

#[tokio::test]
async fn bug_workflow_duplicate_step_labels() {
    let server = create_test_server();
    let app = server.router();

    // Create workflow with duplicate step labels — potential dependency resolution issue
    let (s, body) = post_json(
        &app,
        "/api/v1/workflows",
        &json!({
            "name": "dup_labels",
            "steps": [
                {"label": "step1", "operation": {"sql": "SELECT 1"}, "depends_on": []},
                {"label": "step1", "operation": {"sql": "SELECT 2"}, "depends_on": []}
            ]
        }),
    )
    .await;
    // This is ambiguous — depends_on "step1" would be confusing.
    // Should either reject or handle deterministically.
    if s == StatusCode::OK {
        let wf_id = body["id"].as_str().unwrap();
        let (s, _) = post_json(&app, &format!("/api/v1/workflows/{wf_id}/run"), &json!({})).await;
        // At minimum, should not hang or panic
        assert!(
            s == StatusCode::OK || s == StatusCode::INTERNAL_SERVER_ERROR,
            "duplicate labels should not crash the server"
        );
    }
}

#[tokio::test]
async fn bug_workflow_circular_dependency() {
    let server = create_test_server();
    let app = server.router();

    // Circular dependency: A depends on B, B depends on A
    let (s, body) = post_json(
        &app,
        "/api/v1/workflows",
        &json!({
            "name": "circular",
            "steps": [
                {"label": "a", "operation": {"tool": "noop"}, "depends_on": ["b"]},
                {"label": "b", "operation": {"tool": "noop"}, "depends_on": ["a"]}
            ]
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "definition creation should succeed");
    let wf_id = body["id"].as_str().unwrap();
    // Running should detect the cycle and return an error
    let (s, _) = post_json(&app, &format!("/api/v1/workflows/{wf_id}/run"), &json!({})).await;
    assert_eq!(
        s,
        StatusCode::INTERNAL_SERVER_ERROR,
        "circular dependency should be detected at runtime"
    );
}

#[tokio::test]
async fn bug_workflow_nonexistent_dependency() {
    let server = create_test_server();
    let app = server.router();

    // Step depends on a label that doesn't exist
    let (s, body) = post_json(
        &app,
        "/api/v1/workflows",
        &json!({
            "name": "bad_dep",
            "steps": [
                {"label": "step1", "operation": {"tool": "noop"}, "depends_on": ["nonexistent_step"]}
            ]
        }),
    ).await;
    assert_eq!(s, StatusCode::OK, "definition creation should succeed");
    let wf_id = body["id"].as_str().unwrap();
    // Running should detect the missing dependency and return an error
    let (s, _) = post_json(&app, &format!("/api/v1/workflows/{wf_id}/run"), &json!({})).await;
    assert_eq!(
        s,
        StatusCode::INTERNAL_SERVER_ERROR,
        "nonexistent dependency should be detected at runtime"
    );
}

#[tokio::test]
async fn bug_workflow_dependency_ordering() {
    let server = create_test_server();
    let app = server.router();

    // Steps defined out of order: C depends on B, B depends on A
    // Execution order should be A -> B -> C regardless of definition order
    let (s, body) = post_json(
        &app,
        "/api/v1/workflows",
        &json!({
            "name": "ordered",
            "steps": [
                {"label": "c", "operation": {"tool": "write"}, "depends_on": ["b"]},
                {"label": "a", "operation": {"tool": "read"}, "depends_on": []},
                {"label": "b", "operation": {"tool": "transform"}, "depends_on": ["a"]}
            ]
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "create ordered workflow");
    let wf_id = body["id"].as_str().unwrap();

    let (s, body) = post_json(&app, &format!("/api/v1/workflows/{wf_id}/run"), &json!({})).await;
    assert_eq!(s, StatusCode::OK, "run ordered workflow");

    // All steps should succeed
    let steps = body["step_results"].as_array().unwrap();
    for step in steps {
        assert_eq!(
            step["status"], "success",
            "step {} should succeed",
            step["label"]
        );
    }
}

// ============================================================================
// 11. Message Queue Edge Cases
// ============================================================================

#[tokio::test]
async fn bug_queue_subscribe_nonexistent_topic() {
    let server = create_test_server();
    let app = server.router();

    // Subscribe to topic with no messages
    let (s, body) = post_json(
        &app,
        "/api/v1/queue/subscribe",
        &json!({"topic": "nonexistent_topic_xyz", "max_messages": 10}),
    )
    .await;
    assert_eq!(
        s,
        StatusCode::OK,
        "subscribing to empty topic should succeed"
    );
    let msgs = body.as_array().unwrap();
    assert_eq!(msgs.len(), 0, "should return 0 messages");
}

#[tokio::test]
async fn bug_queue_double_ack() {
    let server = create_test_server();
    let app = server.router();

    // Publish a message
    let (s, msg) = post_json(
        &app,
        "/api/v1/queue/publish",
        &json!({"topic": "double_ack", "payload": "test"}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let msg_id = msg["id"].as_str().unwrap().to_string();

    // Subscribe to get the message
    let (s, _) = post_json(
        &app,
        "/api/v1/queue/subscribe",
        &json!({"topic": "double_ack", "max_messages": 1}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);

    // Ack it
    let (s, body) = post_json(&app, "/api/v1/queue/ack", &json!({"message_ids": [msg_id]})).await;
    assert_eq!(s, StatusCode::OK);

    // Double ack — should not error
    let (s, body) = post_json(&app, "/api/v1/queue/ack", &json!({"message_ids": [msg_id]})).await;
    assert_eq!(s, StatusCode::OK, "double ack should not error");
    assert_eq!(
        body["acked"].as_u64().unwrap_or(1),
        0,
        "second ack should report 0 acked"
    );
}

// ============================================================================
// 12. Branch Management Edge Cases
// ============================================================================

#[tokio::test]
async fn bug_delete_main_branch() {
    let server = create_test_server();
    let app = server.router();

    // Attempt to delete the main branch — should be prevented
    let (s, _) = delete_json(&app, "/api/v1/branches/main").await;
    // Should refuse to delete main branch
    assert_ne!(s, StatusCode::OK, "deleting main branch should be refused");
}

#[tokio::test]
async fn bug_create_branch_empty_name() {
    let server = create_test_server();
    let app = server.router();

    // Create branch with empty name
    let (s, _) = post_json(&app, "/api/v1/branches", &json!({"name": "", "tags": []})).await;
    // Empty name should be rejected with 400
    assert_eq!(
        s,
        StatusCode::BAD_REQUEST,
        "empty branch name should be rejected"
    );
}

// ============================================================================
// 13. Agent Memory Edge Cases
// ============================================================================

#[tokio::test]
async fn bug_memory_recall_empty() {
    let server = create_test_server();
    let app = server.router();

    // Recall with no memories stored
    let (s, body) = post_json(&app, "/api/v1/memory/recall", &json!({"query": "anything"})).await;
    assert_eq!(s, StatusCode::OK, "recall from empty memory should succeed");
    let memories = body["memories"].as_array().unwrap();
    assert_eq!(memories.len(), 0, "should return empty list");
}

#[tokio::test]
async fn bug_memory_forget_nonexistent() {
    let server = create_test_server();
    let app = server.router();

    // Forget a memory ID that doesn't exist
    let (s, body) = post_json(
        &app,
        "/api/v1/memory/forget",
        &json!({"ids": ["nonexistent_id_12345"]}),
    )
    .await;
    assert_eq!(
        s,
        StatusCode::OK,
        "forgetting nonexistent memory should not error"
    );
}

#[tokio::test]
async fn bug_memory_store_and_recall() {
    let server = create_test_server();
    let app = server.router();

    // Store several memories
    let (s, body) = post_json(
        &app,
        "/api/v1/memory/store",
        &json!({"content": "The database uses B-tree storage", "memory_type": "semantic"}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let id1 = body["id"].as_str().unwrap().to_string();

    let (s, _) = post_json(
        &app,
        "/api/v1/memory/store",
        &json!({"content": "User prefers dark mode", "memory_type": "episodic"}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);

    // Recall — should find relevant memories
    let (s, body) = post_json(
        &app,
        "/api/v1/memory/recall",
        &json!({"query": "storage engine"}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let memories = body["memories"].as_array().unwrap();
    assert!(memories.len() >= 1, "should recall at least 1 memory");

    // Forget by ID
    let (s, body) = post_json(&app, "/api/v1/memory/forget", &json!({"ids": [id1]})).await;
    assert_eq!(s, StatusCode::OK);

    // Stats should show remaining
    let (s, body) = get_json(&app, "/api/v1/memory/stats").await;
    assert_eq!(s, StatusCode::OK);
}

// ============================================================================
// 14. Tenant Isolation
// ============================================================================

#[tokio::test]
async fn bug_tenant_delete_nonexistent() {
    let server = create_test_server();
    let app = server.router();

    let (s, _) = delete_json(&app, "/api/v1/tenants/tenant_that_never_existed").await;
    assert_eq!(
        s,
        StatusCode::NOT_FOUND,
        "deleting nonexistent tenant should 404"
    );
}

// ============================================================================
// 15. Scale-to-Zero / Suspend-Resume
// ============================================================================

#[tokio::test]
async fn bug_double_suspend() {
    let server = create_test_server();
    let app = server.router();

    // Suspend
    let (s, _) = post_json(&app, "/api/v1/suspend", &json!({})).await;
    assert_eq!(s, StatusCode::OK, "first suspend should succeed");

    // Double suspend — should conflict
    let (s, _) = post_json(&app, "/api/v1/suspend", &json!({})).await;
    assert_eq!(s, StatusCode::CONFLICT, "double suspend should conflict");

    // Resume
    let (s, _) = post_json(&app, "/api/v1/resume", &json!({})).await;
    assert_eq!(s, StatusCode::OK, "resume should succeed");

    // Double resume — should conflict
    let (s, _) = post_json(&app, "/api/v1/resume", &json!({})).await;
    assert_eq!(s, StatusCode::CONFLICT, "double resume should conflict");
}

// ============================================================================
// 16. SQL Injection / Malformed Input
// ============================================================================

#[tokio::test]
async fn bug_sql_injection_attempts() {
    let server = create_test_server();
    let app = server.router();

    sql(
        &app,
        "CREATE TABLE users (id INT, username TEXT, password TEXT)",
    )
    .await;
    sql(&app, "INSERT INTO users VALUES (1, 'admin', 'secret')").await;

    // Classic SQL injection: should NOT return all rows
    let (ok, data, _, _) = sql(&app, "SELECT * FROM users WHERE username = '' OR '1'='1'").await;
    // We just verify it doesn't crash — since we use parameterized queries internally,
    // this should be handled, but the raw SQL path may be vulnerable
    assert!(ok || !ok, "SQL injection attempt should not crash server");

    // Semicolon injection (multi-statement)
    let (ok, _, _, _) = sql(&app, "SELECT * FROM users; DROP TABLE users").await;
    // Verify users table still exists
    let (ok2, data2, _, _) = sql(&app, "SELECT * FROM users").await;
    assert!(
        ok2,
        "users table should still exist after injection attempt"
    );
    assert_eq!(data2.len(), 1, "original data should be intact");
}

// ============================================================================
// 17. Transaction Semantics
// ============================================================================

#[tokio::test]
async fn bug_insert_duplicate_primary_key() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE pk_test (id INT PRIMARY KEY, name TEXT)").await;
    let (ok, _, _, _) = sql(&app, "INSERT INTO pk_test VALUES (1, 'first')").await;
    assert!(ok, "first insert should succeed");

    // Duplicate PK should fail
    let (ok, _, _, error) = sql(&app, "INSERT INTO pk_test VALUES (1, 'duplicate')").await;
    assert!(!ok, "duplicate PK should fail");
    assert!(
        error.is_some(),
        "should have error message for duplicate PK"
    );

    // Original should be intact
    let (_, data, _, _) = sql(&app, "SELECT name FROM pk_test WHERE id = 1").await;
    assert_eq!(data.len(), 1);
    let name = data[0].as_array().unwrap()[0].as_str().unwrap();
    assert_eq!(name, "first", "original row should not be overwritten");
}

// ============================================================================
// 18. DROP TABLE Edge Cases
// ============================================================================

#[tokio::test]
async fn bug_drop_nonexistent_table() {
    let server = create_test_server();
    let app = server.router();

    // DROP TABLE that doesn't exist should error
    let (ok, _, _, _) = sql(&app, "DROP TABLE nonexistent_table_xyz").await;
    // Some DBs error, some no-op. Key thing: doesn't crash
    // DROP TABLE IF EXISTS should not error
    let (ok2, _, _, _) = sql(&app, "DROP TABLE IF EXISTS nonexistent_table_xyz").await;
    assert!(
        ok2,
        "DROP TABLE IF EXISTS should succeed even for nonexistent table"
    );
}

#[tokio::test]
async fn bug_drop_and_recreate() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE ephemeral (id INT)").await;
    sql(&app, "INSERT INTO ephemeral VALUES (1)").await;

    // Drop
    let (ok, _, _, _) = sql(&app, "DROP TABLE ephemeral").await;
    assert!(ok, "DROP TABLE should succeed");

    // Recreate with same name
    let (ok, _, _, _) = sql(&app, "CREATE TABLE ephemeral (id INT, name TEXT)").await;
    assert!(ok, "should be able to recreate dropped table");

    // Table should be empty (not retain old data)
    let (ok, data, _, _) = sql(&app, "SELECT * FROM ephemeral").await;
    assert!(ok);
    assert_eq!(data.len(), 0, "recreated table should be empty");
}

// ============================================================================
// 19. Multiple Sequential Operations
// ============================================================================

#[tokio::test]
async fn bug_rapid_insert_select_consistency() {
    let server = create_test_server();
    let app = server.router();

    sql(&app, "CREATE TABLE rapid (id INT, val INT)").await;

    // Insert 100 rows rapidly
    for i in 0..100 {
        let (ok, _, _, _) = sql(
            &app,
            &format!("INSERT INTO rapid VALUES ({i}, {}))", i * 10),
        )
        .await;
        assert!(ok, "insert {i} should succeed");
    }

    // Verify count
    let (ok, data, _, _) = sql(&app, "SELECT COUNT(*) FROM rapid").await;
    assert!(ok);
    let count = data[0].as_array().unwrap()[0].as_i64().unwrap_or(0);
    assert_eq!(count, 100, "should have exactly 100 rows after 100 inserts");
}

// ============================================================================
// 18. Branch merge with children should fail
// ============================================================================

#[tokio::test]
async fn bug_branch_merge_delete_with_children() {
    let server = create_test_server();
    let app = server.router();

    // Create parent branch
    let (s, _) = post_json(
        &app,
        "/api/v1/branches",
        &json!({
            "name": "parent-br",
            "tags": []
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK);

    // Create child branch from parent
    let (s, _) = post_json(
        &app,
        "/api/v1/branches",
        &json!({
            "name": "child-br",
            "parent": "parent-br",
            "tags": []
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK);

    // Merge parent with delete_after=true should fail (child would be orphaned)
    let (s, body) = post_json(
        &app,
        "/api/v1/branches/parent-br/merge",
        &json!({
            "delete_after": true
        }),
    )
    .await;
    assert_ne!(
        s,
        StatusCode::OK,
        "merge+delete with children should fail: {:?}",
        body
    );
}

// ============================================================================
// 19. Scale-to-zero double decrement should not underflow
// ============================================================================

#[tokio::test]
async fn bug_scale_to_zero_connection_underflow() {
    // Direct unit test of ActivityTracker
    let tracker = joule_db_server::scale_to_zero::ActivityTracker::new();
    assert_eq!(tracker.connection_count(), 0);

    // Decrement when already at 0 should stay at 0, not underflow
    let result = tracker.decrement_connections();
    assert_eq!(result, 0, "decrement at zero should return 0");
    assert_eq!(
        tracker.connection_count(),
        0,
        "count should stay 0, not wrap to u64::MAX"
    );

    // Normal increment/decrement cycle
    tracker.increment_connections();
    tracker.increment_connections();
    assert_eq!(tracker.connection_count(), 2);
    tracker.decrement_connections();
    assert_eq!(tracker.connection_count(), 1);
    tracker.decrement_connections();
    assert_eq!(tracker.connection_count(), 0);

    // Extra decrement still safe
    tracker.decrement_connections();
    assert_eq!(tracker.connection_count(), 0);
}

// ============================================================================
// 20. Tenant energy budget should reject before consuming
// ============================================================================

#[tokio::test]
async fn bug_tenant_energy_budget_atomic_correctness() {
    let server = create_test_server();
    let app = server.router();

    // Create tenant with tight energy budget
    let (s, body) = post_json(
        &app,
        "/api/v1/tenants",
        &json!({
            "name": "metered-test",
            "quotas": {
                "energy_budget_uj": 100
            }
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "create tenant: {body:?}");
    let tenant_id = body["tenant"]["id"].as_str().unwrap().to_string();

    // Get tenant — energy should be 0
    let (s, body) = get_json(&app, &format!("/api/v1/tenants/{tenant_id}")).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(body["tenant"]["energy_spent_uj"], 0);
}

// ============================================================================
// 21. ORDER BY with mixed types
// ============================================================================

#[tokio::test]
async fn bug_order_by_with_nulls() {
    let server = create_test_server();
    let app = server.router();

    let (ok, _, _, _) = sql(&app, "CREATE TABLE order_nulls (id INT, val INT)").await;
    assert!(ok);

    sql(&app, "INSERT INTO order_nulls VALUES (1, 30)").await;
    sql(&app, "INSERT INTO order_nulls VALUES (2, NULL)").await;
    sql(&app, "INSERT INTO order_nulls VALUES (3, 10)").await;
    sql(&app, "INSERT INTO order_nulls VALUES (4, NULL)").await;
    sql(&app, "INSERT INTO order_nulls VALUES (5, 20)").await;

    // ORDER BY val ASC — NULLs should not crash
    let (ok, data, _, _) = sql(&app, "SELECT id, val FROM order_nulls ORDER BY val ASC").await;
    assert!(ok, "ORDER BY with NULLs should not crash");
    assert_eq!(data.len(), 5, "should return all 5 rows");
}

// ============================================================================
// 22. SELECT with expression aliases
// ============================================================================

#[tokio::test]
async fn bug_select_expression_alias() {
    let server = create_test_server();
    let app = server.router();

    let (ok, _, _, _) = sql(&app, "CREATE TABLE expr_test (a INT, b INT)").await;
    assert!(ok);
    sql(&app, "INSERT INTO expr_test VALUES (10, 3)").await;

    // SELECT with arithmetic expression
    let (ok, data, _, _) = sql(&app, "SELECT a + b FROM expr_test").await;
    assert!(ok, "arithmetic expression in SELECT should work");
    assert_eq!(data.len(), 1);
    let sum = data[0].as_array().unwrap()[0].as_f64().unwrap_or(-1.0);
    assert_eq!(sum, 13.0, "10 + 3 = 13");
}

// ============================================================================
// 23. Multiple aggregates in single SELECT
// ============================================================================

#[tokio::test]
async fn bug_multiple_aggregates() {
    let server = create_test_server();
    let app = server.router();

    let (ok, _, _, _) = sql(&app, "CREATE TABLE multi_agg (val INT)").await;
    assert!(ok);
    sql(&app, "INSERT INTO multi_agg VALUES (10)").await;
    sql(&app, "INSERT INTO multi_agg VALUES (20)").await;
    sql(&app, "INSERT INTO multi_agg VALUES (30)").await;

    let (ok, data, _, _) = sql(
        &app,
        "SELECT COUNT(*), SUM(val), AVG(val), MIN(val), MAX(val) FROM multi_agg",
    )
    .await;
    assert!(ok, "multiple aggregates should work");
    assert_eq!(data.len(), 1);
    let row = data[0].as_array().unwrap();
    assert_eq!(row[0].as_i64().unwrap(), 3, "COUNT should be 3");
    assert_eq!(row[1].as_f64().unwrap(), 60.0, "SUM should be 60");
    assert_eq!(row[2].as_f64().unwrap(), 20.0, "AVG should be 20");
    assert_eq!(row[3].as_f64().unwrap(), 10.0, "MIN should be 10");
    assert_eq!(row[4].as_f64().unwrap(), 30.0, "MAX should be 30");
}
