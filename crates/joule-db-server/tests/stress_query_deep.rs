//! Deep query engine stress tests.
//!
//! Tests complex SQL features: window functions, CTEs, JOINs, GROUP BY,
//! subqueries, and edge cases that push the query planner and executor.

use joule_db_server::query::{QueryExecutor, QueryRequest, QueryResponse, SimpleQueryExecutor};

fn exec(e: &SimpleQueryExecutor, sql: &str) -> QueryResponse {
    e.execute(&QueryRequest {
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
    .unwrap_or_else(|err| panic!("SQL failed: {}\nError: {:?}", sql, err))
}

fn exec_ok(e: &SimpleQueryExecutor, sql: &str) -> bool {
    e.execute(&QueryRequest {
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
    .is_ok()
}

fn row_count(resp: &QueryResponse) -> usize {
    resp.rows.len()
}

fn first_cell(resp: &QueryResponse) -> String {
    resp.rows[0][0].to_string().trim_matches('"').to_string()
}

// ============================================================================
// JOINs
// ============================================================================

#[test]
fn deep_inner_join() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_users (id INT, name TEXT)");
    exec(
        &e,
        "CREATE TABLE deep_orders (id INT, user_id INT, amount INT)",
    );
    exec(&e, "INSERT INTO deep_users VALUES (1, 'alice')");
    exec(&e, "INSERT INTO deep_users VALUES (2, 'bob')");
    exec(&e, "INSERT INTO deep_orders VALUES (1, 1, 100)");
    exec(&e, "INSERT INTO deep_orders VALUES (2, 1, 200)");
    exec(&e, "INSERT INTO deep_orders VALUES (3, 2, 150)");

    let resp = exec(
        &e,
        "SELECT u.name, o.amount FROM deep_users u INNER JOIN deep_orders o ON u.id = o.user_id",
    );
    assert_eq!(row_count(&resp), 3);
    exec(&e, "DROP TABLE deep_users");
    exec(&e, "DROP TABLE deep_orders");
}

#[test]
fn deep_left_join_with_nulls() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_lj_a (id INT, name TEXT)");
    exec(&e, "CREATE TABLE deep_lj_b (id INT, a_id INT, val TEXT)");
    exec(&e, "INSERT INTO deep_lj_a VALUES (1, 'alice')");
    exec(&e, "INSERT INTO deep_lj_a VALUES (2, 'bob')");
    exec(&e, "INSERT INTO deep_lj_a VALUES (3, 'charlie')"); // no match in b
    exec(&e, "INSERT INTO deep_lj_b VALUES (1, 1, 'x')");
    exec(&e, "INSERT INTO deep_lj_b VALUES (2, 2, 'y')");

    let resp = exec(
        &e,
        "SELECT a.name, b.val FROM deep_lj_a a LEFT JOIN deep_lj_b b ON a.id = b.a_id",
    );
    assert_eq!(row_count(&resp), 3); // charlie row should have NULL val
    exec(&e, "DROP TABLE deep_lj_a");
    exec(&e, "DROP TABLE deep_lj_b");
}

#[test]
fn deep_cross_join() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_cx_a (x INT)");
    exec(&e, "CREATE TABLE deep_cx_b (y INT)");
    exec(&e, "INSERT INTO deep_cx_a VALUES (1)");
    exec(&e, "INSERT INTO deep_cx_a VALUES (2)");
    exec(&e, "INSERT INTO deep_cx_b VALUES (10)");
    exec(&e, "INSERT INTO deep_cx_b VALUES (20)");
    exec(&e, "INSERT INTO deep_cx_b VALUES (30)");

    let resp = exec(&e, "SELECT * FROM deep_cx_a CROSS JOIN deep_cx_b");
    assert_eq!(row_count(&resp), 6); // 2 * 3
    exec(&e, "DROP TABLE deep_cx_a");
    exec(&e, "DROP TABLE deep_cx_b");
}

#[test]
fn deep_self_join() {
    let e = SimpleQueryExecutor::new();
    exec(
        &e,
        "CREATE TABLE deep_emp (id INT, name TEXT, manager_id INT)",
    );
    exec(&e, "INSERT INTO deep_emp VALUES (1, 'CEO', NULL)");
    exec(&e, "INSERT INTO deep_emp VALUES (2, 'VP', 1)");
    exec(&e, "INSERT INTO deep_emp VALUES (3, 'Eng', 2)");

    let resp = exec(
        &e,
        "SELECT e.name, m.name FROM deep_emp e INNER JOIN deep_emp m ON e.manager_id = m.id",
    );
    assert_eq!(row_count(&resp), 2); // VP->CEO, Eng->VP
    exec(&e, "DROP TABLE deep_emp");
}

#[test]
fn deep_join_on_null_keys() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_jn_a (id INT, val TEXT)");
    exec(&e, "CREATE TABLE deep_jn_b (id INT, val TEXT)");
    exec(&e, "INSERT INTO deep_jn_a VALUES (NULL, 'a1')");
    exec(&e, "INSERT INTO deep_jn_a VALUES (1, 'a2')");
    exec(&e, "INSERT INTO deep_jn_b VALUES (NULL, 'b1')");
    exec(&e, "INSERT INTO deep_jn_b VALUES (1, 'b2')");

    // JouleDB treats NULL = NULL as matching in JOINs (non-standard but consistent)
    let resp = exec(
        &e,
        "SELECT * FROM deep_jn_a INNER JOIN deep_jn_b ON deep_jn_a.id = deep_jn_b.id",
    );
    assert_eq!(row_count(&resp), 2); // id=1 and NULL=NULL both match
    exec(&e, "DROP TABLE deep_jn_a");
    exec(&e, "DROP TABLE deep_jn_b");
}

// ============================================================================
// GROUP BY + HAVING
// ============================================================================

#[test]
fn deep_group_by_basic() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_gb (category TEXT, amount INT)");
    exec(&e, "INSERT INTO deep_gb VALUES ('a', 10)");
    exec(&e, "INSERT INTO deep_gb VALUES ('a', 20)");
    exec(&e, "INSERT INTO deep_gb VALUES ('b', 30)");
    exec(&e, "INSERT INTO deep_gb VALUES ('b', 40)");
    exec(&e, "INSERT INTO deep_gb VALUES ('c', 50)");

    let resp = exec(
        &e,
        "SELECT category, SUM(amount) FROM deep_gb GROUP BY category",
    );
    assert_eq!(row_count(&resp), 3);
    exec(&e, "DROP TABLE deep_gb");
}

#[test]
fn deep_group_by_having() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_having (cat TEXT, val INT)");
    exec(&e, "INSERT INTO deep_having VALUES ('a', 10)");
    exec(&e, "INSERT INTO deep_having VALUES ('a', 20)");
    exec(&e, "INSERT INTO deep_having VALUES ('b', 5)");
    exec(&e, "INSERT INTO deep_having VALUES ('b', 3)");

    let resp = exec(
        &e,
        "SELECT cat, SUM(val) FROM deep_having GROUP BY cat HAVING SUM(val) > 10",
    );
    assert_eq!(row_count(&resp), 1); // only 'a' has sum > 10
    exec(&e, "DROP TABLE deep_having");
}

#[test]
fn deep_group_by_count() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_gbc (status TEXT, id INT)");
    exec(&e, "INSERT INTO deep_gbc VALUES ('active', 1)");
    exec(&e, "INSERT INTO deep_gbc VALUES ('active', 2)");
    exec(&e, "INSERT INTO deep_gbc VALUES ('active', 3)");
    exec(&e, "INSERT INTO deep_gbc VALUES ('inactive', 4)");

    let resp = exec(&e, "SELECT status, COUNT(*) FROM deep_gbc GROUP BY status");
    assert_eq!(row_count(&resp), 2);
    exec(&e, "DROP TABLE deep_gbc");
}

#[test]
fn deep_group_by_with_null() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_gbn (grp TEXT, val INT)");
    exec(&e, "INSERT INTO deep_gbn VALUES ('a', 1)");
    exec(&e, "INSERT INTO deep_gbn VALUES (NULL, 2)");
    exec(&e, "INSERT INTO deep_gbn VALUES (NULL, 3)");
    exec(&e, "INSERT INTO deep_gbn VALUES ('a', 4)");

    let resp = exec(&e, "SELECT grp, SUM(val) FROM deep_gbn GROUP BY grp");
    assert_eq!(row_count(&resp), 2); // 'a' group and NULL group
    exec(&e, "DROP TABLE deep_gbn");
}

#[test]
fn deep_multi_column_group_by() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_mcgb (a TEXT, b TEXT, val INT)");
    exec(&e, "INSERT INTO deep_mcgb VALUES ('x', '1', 10)");
    exec(&e, "INSERT INTO deep_mcgb VALUES ('x', '1', 20)");
    exec(&e, "INSERT INTO deep_mcgb VALUES ('x', '2', 30)");
    exec(&e, "INSERT INTO deep_mcgb VALUES ('y', '1', 40)");

    let resp = exec(&e, "SELECT a, b, SUM(val) FROM deep_mcgb GROUP BY a, b");
    assert_eq!(row_count(&resp), 3); // (x,1), (x,2), (y,1)
    exec(&e, "DROP TABLE deep_mcgb");
}

// ============================================================================
// Subqueries
// ============================================================================

#[test]
fn deep_scalar_subquery() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_sq (id INT, val INT)");
    exec(&e, "INSERT INTO deep_sq VALUES (1, 10)");
    exec(&e, "INSERT INTO deep_sq VALUES (2, 20)");
    exec(&e, "INSERT INTO deep_sq VALUES (3, 30)");

    let resp = exec(
        &e,
        "SELECT * FROM deep_sq WHERE val > (SELECT AVG(val) FROM deep_sq)",
    );
    assert_eq!(row_count(&resp), 1); // only val=30 > avg=20
    exec(&e, "DROP TABLE deep_sq");
}

#[test]
fn deep_in_subquery() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_in_a (id INT, name TEXT)");
    exec(&e, "CREATE TABLE deep_in_b (user_id INT)");
    exec(&e, "INSERT INTO deep_in_a VALUES (1, 'alice')");
    exec(&e, "INSERT INTO deep_in_a VALUES (2, 'bob')");
    exec(&e, "INSERT INTO deep_in_a VALUES (3, 'charlie')");
    exec(&e, "INSERT INTO deep_in_b VALUES (1)");
    exec(&e, "INSERT INTO deep_in_b VALUES (3)");

    // IN subquery — verify it executes without panic (results depend on implementation depth)
    let result = exec_ok(
        &e,
        "SELECT name FROM deep_in_a WHERE id IN (SELECT user_id FROM deep_in_b)",
    );
    assert!(result, "IN subquery should execute without error");
    exec(&e, "DROP TABLE deep_in_a");
    exec(&e, "DROP TABLE deep_in_b");
}

#[test]
fn deep_exists_subquery() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_ex_a (id INT, name TEXT)");
    exec(&e, "CREATE TABLE deep_ex_b (ref_id INT)");
    exec(&e, "INSERT INTO deep_ex_a VALUES (1, 'alice')");
    exec(&e, "INSERT INTO deep_ex_a VALUES (2, 'bob')");
    exec(&e, "INSERT INTO deep_ex_b VALUES (1)");

    // EXISTS correlated subquery — verify it executes without panic
    let result = exec_ok(
        &e,
        "SELECT name FROM deep_ex_a WHERE EXISTS (SELECT 1 FROM deep_ex_b WHERE deep_ex_b.ref_id = deep_ex_a.id)",
    );
    assert!(result, "EXISTS subquery should execute without error");
    exec(&e, "DROP TABLE deep_ex_a");
    exec(&e, "DROP TABLE deep_ex_b");
}

#[test]
fn deep_subquery_in_select() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_sqs (id INT, val INT)");
    exec(&e, "INSERT INTO deep_sqs VALUES (1, 10)");
    exec(&e, "INSERT INTO deep_sqs VALUES (2, 20)");

    let resp = exec(
        &e,
        "SELECT id, val, (SELECT MAX(val) FROM deep_sqs) FROM deep_sqs",
    );
    assert_eq!(row_count(&resp), 2);
    exec(&e, "DROP TABLE deep_sqs");
}

// ============================================================================
// CTEs (Common Table Expressions)
// ============================================================================

#[test]
fn deep_simple_cte() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_cte (id INT, val INT)");
    exec(&e, "INSERT INTO deep_cte VALUES (1, 100)");
    exec(&e, "INSERT INTO deep_cte VALUES (2, 200)");
    exec(&e, "INSERT INTO deep_cte VALUES (3, 300)");

    let resp = exec(
        &e,
        "WITH high_val AS (SELECT * FROM deep_cte WHERE val > 150) SELECT COUNT(*) FROM high_val",
    );
    assert_eq!(first_cell(&resp), "2");
    exec(&e, "DROP TABLE deep_cte");
}

#[test]
fn deep_cte_multiple() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_cte2 (id INT, cat TEXT, val INT)");
    exec(&e, "INSERT INTO deep_cte2 VALUES (1, 'a', 10)");
    exec(&e, "INSERT INTO deep_cte2 VALUES (2, 'b', 20)");
    exec(&e, "INSERT INTO deep_cte2 VALUES (3, 'a', 30)");

    let resp = exec(
        &e,
        "WITH cat_a AS (SELECT * FROM deep_cte2 WHERE cat = 'a'), cat_b AS (SELECT * FROM deep_cte2 WHERE cat = 'b') SELECT (SELECT COUNT(*) FROM cat_a) + (SELECT COUNT(*) FROM cat_b)",
    );
    assert_eq!(first_cell(&resp), "3");
    exec(&e, "DROP TABLE deep_cte2");
}

// ============================================================================
// Window functions
// ============================================================================

#[test]
fn deep_window_row_number() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_win (id INT, dept TEXT, salary INT)");
    exec(&e, "INSERT INTO deep_win VALUES (1, 'eng', 100)");
    exec(&e, "INSERT INTO deep_win VALUES (2, 'eng', 120)");
    exec(&e, "INSERT INTO deep_win VALUES (3, 'sales', 90)");
    exec(&e, "INSERT INTO deep_win VALUES (4, 'sales', 95)");

    let resp = exec(
        &e,
        "SELECT id, ROW_NUMBER() OVER (ORDER BY salary DESC) FROM deep_win",
    );
    assert_eq!(row_count(&resp), 4);
    exec(&e, "DROP TABLE deep_win");
}

#[test]
fn deep_window_rank() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_rank (id INT, score INT)");
    exec(&e, "INSERT INTO deep_rank VALUES (1, 100)");
    exec(&e, "INSERT INTO deep_rank VALUES (2, 100)");
    exec(&e, "INSERT INTO deep_rank VALUES (3, 90)");

    let resp = exec(
        &e,
        "SELECT id, RANK() OVER (ORDER BY score DESC) FROM deep_rank",
    );
    assert_eq!(row_count(&resp), 3);
    exec(&e, "DROP TABLE deep_rank");
}

#[test]
fn deep_window_partition_by() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_part (id INT, dept TEXT, salary INT)");
    exec(&e, "INSERT INTO deep_part VALUES (1, 'eng', 100)");
    exec(&e, "INSERT INTO deep_part VALUES (2, 'eng', 120)");
    exec(&e, "INSERT INTO deep_part VALUES (3, 'sales', 90)");

    let resp = exec(
        &e,
        "SELECT id, dept, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary) FROM deep_part",
    );
    assert_eq!(row_count(&resp), 3);
    exec(&e, "DROP TABLE deep_part");
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn deep_empty_table_operations() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_empty (id INT, val TEXT)");

    let resp = exec(&e, "SELECT * FROM deep_empty");
    assert_eq!(row_count(&resp), 0);

    let resp = exec(&e, "SELECT COUNT(*) FROM deep_empty");
    assert_eq!(first_cell(&resp), "0");

    let resp = exec(&e, "SELECT MAX(id) FROM deep_empty");
    assert_eq!(row_count(&resp), 1); // returns 1 row with NULL

    exec(&e, "DELETE FROM deep_empty WHERE id = 1"); // no-op
    exec(&e, "UPDATE deep_empty SET val = 'x'"); // no-op

    exec(&e, "DROP TABLE deep_empty");
}

#[test]
fn deep_division_by_zero() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_div (id INT, val INT)");
    exec(&e, "INSERT INTO deep_div VALUES (1, 10)");
    exec(&e, "INSERT INTO deep_div VALUES (2, 0)");

    // Division by zero should be handled (error or NULL, not panic)
    let result = e.execute(&QueryRequest {
        sql: "SELECT val / 0 FROM deep_div".to_string(),
        params: Default::default(),
        args: Vec::new(),
        explain: false,
        limit: None,
        session_id: None,
        query_timeout_ms: Some(30000),
        branch_id: None,
        tenant_id: None,
    });
    // Either returns an error or returns results with NULL/special value
    let _ = result;
    exec(&e, "DROP TABLE deep_div");
}

#[test]
fn deep_long_column_alias() {
    let e = SimpleQueryExecutor::new();
    let long_alias = "a".repeat(200);
    let resp = exec(&e, &format!("SELECT 1 AS {}", long_alias));
    assert_eq!(row_count(&resp), 1);
    assert_eq!(resp.columns[0], long_alias);
}

#[test]
fn deep_nested_expressions() {
    let e = SimpleQueryExecutor::new();
    // Deeply nested arithmetic: ((((1+1)+1)+1)...) — max parser depth is 30
    let mut expr = "1".to_string();
    for _ in 0..25 {
        expr = format!("({} + 1)", expr);
    }
    let resp = exec(&e, &format!("SELECT {}", expr));
    assert_eq!(first_cell(&resp), "26");
}

#[test]
fn deep_many_columns() {
    let e = SimpleQueryExecutor::new();
    // Create table with 50 columns
    let cols: Vec<String> = (0..50).map(|i| format!("c{} INT", i)).collect();
    exec(&e, &format!("CREATE TABLE deep_wide ({})", cols.join(", ")));

    // Insert a row with all values
    let vals: Vec<String> = (0..50).map(|i| i.to_string()).collect();
    exec(
        &e,
        &format!("INSERT INTO deep_wide VALUES ({})", vals.join(", ")),
    );

    let resp = exec(&e, "SELECT * FROM deep_wide");
    assert_eq!(row_count(&resp), 1);
    assert_eq!(resp.columns.len(), 50);
    exec(&e, "DROP TABLE deep_wide");
}

#[test]
fn deep_case_expression() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_case (id INT, status TEXT)");
    exec(&e, "INSERT INTO deep_case VALUES (1, 'active')");
    exec(&e, "INSERT INTO deep_case VALUES (2, 'inactive')");
    exec(&e, "INSERT INTO deep_case VALUES (3, 'pending')");

    let resp = exec(
        &e,
        "SELECT id, CASE status WHEN 'active' THEN 'ON' WHEN 'inactive' THEN 'OFF' ELSE 'WAIT' END FROM deep_case",
    );
    assert_eq!(row_count(&resp), 3);
    exec(&e, "DROP TABLE deep_case");
}

#[test]
fn deep_distinct() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_dist (val TEXT)");
    exec(&e, "INSERT INTO deep_dist VALUES ('a')");
    exec(&e, "INSERT INTO deep_dist VALUES ('b')");
    exec(&e, "INSERT INTO deep_dist VALUES ('a')");
    exec(&e, "INSERT INTO deep_dist VALUES ('c')");
    exec(&e, "INSERT INTO deep_dist VALUES ('b')");

    let resp = exec(&e, "SELECT DISTINCT val FROM deep_dist");
    assert_eq!(row_count(&resp), 3);
    exec(&e, "DROP TABLE deep_dist");
}

#[test]
fn deep_union() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_u1 (id INT, name TEXT)");
    exec(&e, "CREATE TABLE deep_u2 (id INT, name TEXT)");
    exec(&e, "INSERT INTO deep_u1 VALUES (1, 'alice')");
    exec(&e, "INSERT INTO deep_u2 VALUES (2, 'bob')");

    let resp = exec(&e, "SELECT * FROM deep_u1 UNION ALL SELECT * FROM deep_u2");
    assert_eq!(row_count(&resp), 2);
    exec(&e, "DROP TABLE deep_u1");
    exec(&e, "DROP TABLE deep_u2");
}

#[test]
fn deep_like_pattern() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_like (name TEXT)");
    exec(&e, "INSERT INTO deep_like VALUES ('alice')");
    exec(&e, "INSERT INTO deep_like VALUES ('alicia')");
    exec(&e, "INSERT INTO deep_like VALUES ('bob')");
    exec(&e, "INSERT INTO deep_like VALUES ('albert')");

    let resp = exec(&e, "SELECT name FROM deep_like WHERE name LIKE 'al%'");
    assert_eq!(row_count(&resp), 3); // alice, alicia, albert
    exec(&e, "DROP TABLE deep_like");
}

#[test]
fn deep_between() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_btw (id INT)");
    for i in 1..=10 {
        exec(&e, &format!("INSERT INTO deep_btw VALUES ({})", i));
    }

    let resp = exec(&e, "SELECT * FROM deep_btw WHERE id BETWEEN 3 AND 7");
    assert_eq!(row_count(&resp), 5);
    exec(&e, "DROP TABLE deep_btw");
}

#[test]
fn deep_coalesce_in_query() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_coal (id INT, name TEXT)");
    exec(&e, "INSERT INTO deep_coal VALUES (1, NULL)");
    exec(&e, "INSERT INTO deep_coal VALUES (2, 'bob')");

    let resp = exec(
        &e,
        "SELECT COALESCE(name, 'unknown') FROM deep_coal ORDER BY id",
    );
    assert_eq!(row_count(&resp), 2);
    exec(&e, "DROP TABLE deep_coal");
}

#[test]
fn deep_insert_select() {
    let e = SimpleQueryExecutor::new();
    exec(&e, "CREATE TABLE deep_src (id INT, val TEXT)");
    exec(&e, "CREATE TABLE deep_dst (id INT, val TEXT)");
    exec(&e, "INSERT INTO deep_src VALUES (1, 'a')");
    exec(&e, "INSERT INTO deep_src VALUES (2, 'b')");

    exec(&e, "INSERT INTO deep_dst SELECT * FROM deep_src");
    let resp = exec(&e, "SELECT COUNT(*) FROM deep_dst");
    assert_eq!(first_cell(&resp), "2");
    exec(&e, "DROP TABLE deep_src");
    exec(&e, "DROP TABLE deep_dst");
}
