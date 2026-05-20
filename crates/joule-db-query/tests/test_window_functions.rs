//! Tests for Window Function support in JouleDB

use joule_db_query::{
    Expression, QueryPlanner, QueryResult, RowData, SqlParser, SqlStatement, StorageExecutor,
    TableStorage, Value,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// Mock Storage (same as in test_ctes.rs)
struct MemoryTableStorage {
    tables: RwLock<HashMap<String, (Vec<String>, Vec<RowData>)>>,
}

impl MemoryTableStorage {
    fn new() -> Self {
        Self {
            tables: RwLock::new(HashMap::new()),
        }
    }

    fn create_table(&self, name: &str, columns: Vec<String>) {
        let mut tables = self.tables.write().unwrap();
        tables.insert(name.to_string(), (columns, Vec::new()));
    }

    fn insert_row(&self, table: &str, values: Vec<Value>) {
        let mut tables = self.tables.write().unwrap();
        if let Some((cols, rows)) = tables.get_mut(table) {
            rows.push(RowData::new(cols.clone(), values));
        }
    }
}

impl TableStorage for MemoryTableStorage {
    fn scan(&self, table: &str) -> QueryResult<Vec<RowData>> {
        let tables = self.tables.read().unwrap();
        if let Some((_, rows)) = tables.get(table) {
            Ok(rows.clone())
        } else {
            Err(joule_db_query::QueryError::UnknownTable(table.to_string()))
        }
    }

    fn columns(&self, table: &str) -> QueryResult<Vec<String>> {
        let tables = self.tables.read().unwrap();
        if let Some((cols, _)) = tables.get(table) {
            Ok(cols.clone())
        } else {
            Err(joule_db_query::QueryError::UnknownTable(table.to_string()))
        }
    }

    fn insert(&self, _table: &str, _row: &RowData) -> QueryResult<()> {
        Ok(())
    }
    fn update(
        &self,
        _table: &str,
        _assignments: &HashMap<String, Value>,
        _predicate: Option<&Expression>,
    ) -> QueryResult<usize> {
        Ok(0)
    }
    fn delete(&self, _table: &str, _predicate: Option<&Expression>) -> QueryResult<usize> {
        Ok(0)
    }
    fn table_exists(&self, table: &str) -> QueryResult<bool> {
        Ok(self.tables.read().unwrap().contains_key(table))
    }
}

// ============================================================================
// Window Function Parsing Tests
// ============================================================================

#[test]
fn test_parse_row_number() {
    let mut parser = SqlParser::new();
    let result = parser.parse("SELECT name, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM users");
    assert!(result.is_ok(), "Failed to parse ROW_NUMBER: {:?}", result);
}

#[test]
fn test_parse_row_number_with_partition() {
    let mut parser = SqlParser::new();
    let result = parser.parse(
        "SELECT name, ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rn FROM employees"
    );
    assert!(
        result.is_ok(),
        "Failed to parse ROW_NUMBER with PARTITION BY: {:?}",
        result
    );
}

#[test]
fn test_parse_aggregate_window_function() {
    let mut parser = SqlParser::new();
    let result = parser.parse(
        "SELECT name, SUM(salary) OVER (PARTITION BY department) AS dept_total FROM employees",
    );
    assert!(
        result.is_ok(),
        "Failed to parse aggregate window function: {:?}",
        result
    );
}

#[test]
fn test_parse_multiple_window_functions() {
    let mut parser = SqlParser::new();
    let result = parser.parse(
        "SELECT name, 
                ROW_NUMBER() OVER (ORDER BY id) AS rn,
                SUM(amount) OVER (PARTITION BY category) AS category_total
         FROM orders",
    );
    assert!(
        result.is_ok(),
        "Failed to parse multiple window functions: {:?}",
        result
    );
}

// ============================================================================
// Window Function Execution Tests
// ============================================================================

#[test]
fn test_execute_row_number_simple() {
    let storage = Arc::new(MemoryTableStorage::new());
    storage.create_table("users", vec!["id".to_string(), "name".to_string()]);
    storage.insert_row(
        "users",
        vec![Value::Int(1), Value::String("Alice".to_string())],
    );
    storage.insert_row(
        "users",
        vec![Value::Int(2), Value::String("Bob".to_string())],
    );
    storage.insert_row(
        "users",
        vec![Value::Int(3), Value::String("Charlie".to_string())],
    );

    let mut parser = SqlParser::new();
    let stmt = parser
        .parse("SELECT name, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM users")
        .expect("Failed to parse");

    let query = match stmt {
        SqlStatement::Select(q) => q.to_query(),
        _ => panic!("Expected SELECT"),
    };

    let planner = QueryPlanner::new();
    let plan = planner.plan(&query).expect("Failed to plan query");

    let executor = StorageExecutor::new(storage);
    let context = joule_db_query::QueryContext::new();
    let result = executor.execute(&plan, &context);

    // Should execute without error (actual window function computation may vary)
    assert!(
        result.is_ok(),
        "Window function execution failed: {:?}",
        result
    );
    let result = result.unwrap();

    // Should have 3 rows
    assert_eq!(result.rows.len(), 3, "Expected 3 rows");
}

#[test]
fn test_execute_sum_window_with_partition() {
    let storage = Arc::new(MemoryTableStorage::new());
    storage.create_table(
        "sales",
        vec![
            "region".to_string(),
            "product".to_string(),
            "amount".to_string(),
        ],
    );
    // Region A sales
    storage.insert_row(
        "sales",
        vec![
            Value::String("A".to_string()),
            Value::String("Widget".to_string()),
            Value::Int(100),
        ],
    );
    storage.insert_row(
        "sales",
        vec![
            Value::String("A".to_string()),
            Value::String("Gadget".to_string()),
            Value::Int(200),
        ],
    );
    // Region B sales
    storage.insert_row(
        "sales",
        vec![
            Value::String("B".to_string()),
            Value::String("Widget".to_string()),
            Value::Int(150),
        ],
    );

    let mut parser = SqlParser::new();
    let stmt = parser.parse(
        "SELECT region, product, amount, SUM(amount) OVER (PARTITION BY region) AS region_total FROM sales"
    ).expect("Failed to parse");

    let query = match stmt {
        SqlStatement::Select(q) => q.to_query(),
        _ => panic!("Expected SELECT"),
    };

    let planner = QueryPlanner::new();
    let plan = planner.plan(&query).expect("Failed to plan query");

    let executor = StorageExecutor::new(storage);
    let context = joule_db_query::QueryContext::new();
    let result = executor.execute(&plan, &context);

    assert!(
        result.is_ok(),
        "Partitioned SUM window function failed: {:?}",
        result
    );
    let result = result.unwrap();

    // Should have 3 rows
    assert_eq!(result.rows.len(), 3, "Expected 3 rows");
}

#[test]
fn test_execute_count_window() {
    let storage = Arc::new(MemoryTableStorage::new());
    storage.create_table("items", vec!["category".to_string(), "name".to_string()]);
    storage.insert_row(
        "items",
        vec![
            Value::String("A".to_string()),
            Value::String("Item1".to_string()),
        ],
    );
    storage.insert_row(
        "items",
        vec![
            Value::String("A".to_string()),
            Value::String("Item2".to_string()),
        ],
    );
    storage.insert_row(
        "items",
        vec![
            Value::String("B".to_string()),
            Value::String("Item3".to_string()),
        ],
    );

    let mut parser = SqlParser::new();
    let stmt = parser
        .parse(
            "SELECT category, name, COUNT(*) OVER (PARTITION BY category) AS cat_count FROM items",
        )
        .expect("Failed to parse");

    let query = match stmt {
        SqlStatement::Select(q) => q.to_query(),
        _ => panic!("Expected SELECT"),
    };

    let planner = QueryPlanner::new();
    let plan = planner.plan(&query).expect("Failed to plan query");

    let executor = StorageExecutor::new(storage);
    let context = joule_db_query::QueryContext::new();
    let result = executor.execute(&plan, &context);

    assert!(result.is_ok(), "COUNT window function failed: {:?}", result);
    let result = result.unwrap();

    assert_eq!(result.rows.len(), 3, "Expected 3 rows");
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_window_function_no_partition() {
    let storage = Arc::new(MemoryTableStorage::new());
    storage.create_table("numbers", vec!["val".to_string()]);
    storage.insert_row("numbers", vec![Value::Int(10)]);
    storage.insert_row("numbers", vec![Value::Int(20)]);
    storage.insert_row("numbers", vec![Value::Int(30)]);

    let mut parser = SqlParser::new();
    let stmt = parser
        .parse("SELECT val, SUM(val) OVER () AS total FROM numbers")
        .expect("Failed to parse");

    let query = match stmt {
        SqlStatement::Select(q) => q.to_query(),
        _ => panic!("Expected SELECT"),
    };

    let planner = QueryPlanner::new();
    let plan = planner.plan(&query).expect("Failed to plan query");

    let executor = StorageExecutor::new(storage);
    let context = joule_db_query::QueryContext::new();
    let result = executor.execute(&plan, &context);

    assert!(
        result.is_ok(),
        "Window function without PARTITION BY failed: {:?}",
        result
    );
    let result = result.unwrap();

    assert_eq!(result.rows.len(), 3, "Expected 3 rows");
}

#[test]
fn test_window_function_empty_table() {
    let storage = Arc::new(MemoryTableStorage::new());
    storage.create_table("empty_table", vec!["id".to_string(), "name".to_string()]);
    // No rows inserted

    let mut parser = SqlParser::new();
    let stmt = parser
        .parse("SELECT name, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM empty_table")
        .expect("Failed to parse");

    let query = match stmt {
        SqlStatement::Select(q) => q.to_query(),
        _ => panic!("Expected SELECT"),
    };

    let planner = QueryPlanner::new();
    let plan = planner.plan(&query).expect("Failed to plan query");

    let executor = StorageExecutor::new(storage);
    let context = joule_db_query::QueryContext::new();
    let result = executor.execute(&plan, &context);

    assert!(
        result.is_ok(),
        "Window function on empty table failed: {:?}",
        result
    );
    let result = result.unwrap();

    assert_eq!(result.rows.len(), 0, "Expected 0 rows for empty table");
}

// ============================================================================
// Window Function Value Verification Tests (planner-generated Window nodes)
// ============================================================================

#[test]
fn test_window_row_number_values() {
    let storage = Arc::new(MemoryTableStorage::new());
    storage.create_table("users", vec!["id".to_string(), "name".to_string()]);
    storage.insert_row(
        "users",
        vec![Value::Int(3), Value::String("Charlie".to_string())],
    );
    storage.insert_row(
        "users",
        vec![Value::Int(1), Value::String("Alice".to_string())],
    );
    storage.insert_row(
        "users",
        vec![Value::Int(2), Value::String("Bob".to_string())],
    );

    let mut parser = SqlParser::new();
    let stmt = parser
        .parse("SELECT name, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM users")
        .expect("Failed to parse");

    let query = match stmt {
        SqlStatement::Select(q) => q.to_query(),
        _ => panic!("Expected SELECT"),
    };

    let planner = QueryPlanner::new();
    let plan = planner.plan(&query).expect("Failed to plan query");

    let executor = StorageExecutor::new(storage);
    let context = joule_db_query::QueryContext::new();
    let result = executor.execute(&plan, &context).unwrap();

    assert_eq!(result.rows.len(), 3);
    // Rows should be ordered by id, with ROW_NUMBER 1,2,3
    // Check that the rn column exists with sequential values
    let rn_idx = result
        .columns
        .iter()
        .position(|c| c == "rn")
        .expect("rn column not found");
    let rn_values: Vec<_> = result.rows.iter().map(|r| &r.values[rn_idx]).collect();
    assert_eq!(
        rn_values,
        vec![&Value::Int(1), &Value::Int(2), &Value::Int(3)]
    );
}

#[test]
fn test_window_sum_partition_values() {
    let storage = Arc::new(MemoryTableStorage::new());
    storage.create_table(
        "sales",
        vec![
            "region".to_string(),
            "product".to_string(),
            "amount".to_string(),
        ],
    );
    storage.insert_row(
        "sales",
        vec![
            Value::String("A".to_string()),
            Value::String("Widget".to_string()),
            Value::Int(100),
        ],
    );
    storage.insert_row(
        "sales",
        vec![
            Value::String("A".to_string()),
            Value::String("Gadget".to_string()),
            Value::Int(200),
        ],
    );
    storage.insert_row(
        "sales",
        vec![
            Value::String("B".to_string()),
            Value::String("Widget".to_string()),
            Value::Int(150),
        ],
    );

    let mut parser = SqlParser::new();
    let stmt = parser.parse(
        "SELECT region, product, amount, SUM(amount) OVER (PARTITION BY region) AS region_total FROM sales"
    ).expect("Failed to parse");

    let query = match stmt {
        SqlStatement::Select(q) => q.to_query(),
        _ => panic!("Expected SELECT"),
    };

    let planner = QueryPlanner::new();
    let plan = planner.plan(&query).expect("Failed to plan query");

    let executor = StorageExecutor::new(storage);
    let context = joule_db_query::QueryContext::new();
    let result = executor.execute(&plan, &context).unwrap();

    assert_eq!(result.rows.len(), 3);
    // Check region_total column
    let rt_idx = result
        .columns
        .iter()
        .position(|c| c == "region_total")
        .expect("region_total not found");
    // Region A rows should have total 300, Region B should have 150
    for row in &result.rows {
        let region_idx = result.columns.iter().position(|c| c == "region").unwrap();
        match &row.values[region_idx] {
            Value::String(r) if r == "A" => assert_eq!(row.values[rt_idx], Value::Float(300.0)),
            Value::String(r) if r == "B" => assert_eq!(row.values[rt_idx], Value::Float(150.0)),
            _ => {}
        }
    }
}

#[test]
fn test_window_count_partition_values() {
    let storage = Arc::new(MemoryTableStorage::new());
    storage.create_table("items", vec!["category".to_string(), "name".to_string()]);
    storage.insert_row(
        "items",
        vec![
            Value::String("A".to_string()),
            Value::String("Item1".to_string()),
        ],
    );
    storage.insert_row(
        "items",
        vec![
            Value::String("A".to_string()),
            Value::String("Item2".to_string()),
        ],
    );
    storage.insert_row(
        "items",
        vec![
            Value::String("B".to_string()),
            Value::String("Item3".to_string()),
        ],
    );

    let mut parser = SqlParser::new();
    let stmt = parser
        .parse(
            "SELECT category, name, COUNT(*) OVER (PARTITION BY category) AS cat_count FROM items",
        )
        .expect("Failed to parse");

    let query = match stmt {
        SqlStatement::Select(q) => q.to_query(),
        _ => panic!("Expected SELECT"),
    };

    let planner = QueryPlanner::new();
    let plan = planner.plan(&query).expect("Failed to plan query");

    let executor = StorageExecutor::new(storage);
    let context = joule_db_query::QueryContext::new();
    let result = executor.execute(&plan, &context).unwrap();

    assert_eq!(result.rows.len(), 3);
    let cc_idx = result
        .columns
        .iter()
        .position(|c| c == "cat_count")
        .expect("cat_count not found");
    let cat_idx = result.columns.iter().position(|c| c == "category").unwrap();
    for row in &result.rows {
        match &row.values[cat_idx] {
            Value::String(c) if c == "A" => assert_eq!(row.values[cc_idx], Value::Int(2)),
            Value::String(c) if c == "B" => assert_eq!(row.values[cc_idx], Value::Int(1)),
            _ => {}
        }
    }
}

#[test]
fn test_window_rank_values() {
    let storage = Arc::new(MemoryTableStorage::new());
    storage.create_table("scores", vec!["name".to_string(), "score".to_string()]);
    storage.insert_row(
        "scores",
        vec![Value::String("Alice".to_string()), Value::Int(90)],
    );
    storage.insert_row(
        "scores",
        vec![Value::String("Bob".to_string()), Value::Int(90)],
    );
    storage.insert_row(
        "scores",
        vec![Value::String("Charlie".to_string()), Value::Int(80)],
    );

    let mut parser = SqlParser::new();
    let stmt = parser
        .parse("SELECT name, score, RANK() OVER (ORDER BY score DESC) AS rnk FROM scores")
        .expect("Failed to parse");

    let query = match stmt {
        SqlStatement::Select(q) => q.to_query(),
        _ => panic!("Expected SELECT"),
    };

    let planner = QueryPlanner::new();
    let plan = planner.plan(&query).expect("Failed to plan query");

    let executor = StorageExecutor::new(storage);
    let context = joule_db_query::QueryContext::new();
    let result = executor.execute(&plan, &context).unwrap();

    assert_eq!(result.rows.len(), 3);
    let rnk_idx = result
        .columns
        .iter()
        .position(|c| c == "rnk")
        .expect("rnk not found");
    // Two score=90 tie at rank 1, then score=80 is rank 3
    let ranks: Vec<_> = result.rows.iter().map(|r| &r.values[rnk_idx]).collect();
    assert_eq!(ranks, vec![&Value::Int(1), &Value::Int(1), &Value::Int(3)]);
}

#[test]
fn test_window_dense_rank_values() {
    let storage = Arc::new(MemoryTableStorage::new());
    storage.create_table("scores2", vec!["name".to_string(), "score".to_string()]);
    storage.insert_row(
        "scores2",
        vec![Value::String("Alice".to_string()), Value::Int(90)],
    );
    storage.insert_row(
        "scores2",
        vec![Value::String("Bob".to_string()), Value::Int(90)],
    );
    storage.insert_row(
        "scores2",
        vec![Value::String("Charlie".to_string()), Value::Int(80)],
    );

    let mut parser = SqlParser::new();
    let stmt = parser
        .parse("SELECT name, score, DENSE_RANK() OVER (ORDER BY score DESC) AS drnk FROM scores2")
        .expect("Failed to parse");

    let query = match stmt {
        SqlStatement::Select(q) => q.to_query(),
        _ => panic!("Expected SELECT"),
    };

    let planner = QueryPlanner::new();
    let plan = planner.plan(&query).expect("Failed to plan query");

    let executor = StorageExecutor::new(storage);
    let context = joule_db_query::QueryContext::new();
    let result = executor.execute(&plan, &context).unwrap();

    assert_eq!(result.rows.len(), 3);
    let drnk_idx = result
        .columns
        .iter()
        .position(|c| c == "drnk")
        .expect("drnk not found");
    // Dense rank: two 90s at 1, then 80 at 2 (not 3)
    let ranks: Vec<_> = result.rows.iter().map(|r| &r.values[drnk_idx]).collect();
    assert_eq!(ranks, vec![&Value::Int(1), &Value::Int(1), &Value::Int(2)]);
}

#[test]
fn test_window_ntile_planner() {
    let storage = Arc::new(MemoryTableStorage::new());
    storage.create_table("items2", vec!["id".to_string(), "val".to_string()]);
    storage.insert_row(
        "items2",
        vec![Value::Int(1), Value::String("a".to_string())],
    );
    storage.insert_row(
        "items2",
        vec![Value::Int(2), Value::String("b".to_string())],
    );
    storage.insert_row(
        "items2",
        vec![Value::Int(3), Value::String("c".to_string())],
    );
    storage.insert_row(
        "items2",
        vec![Value::Int(4), Value::String("d".to_string())],
    );

    let mut parser = SqlParser::new();
    let stmt = parser
        .parse("SELECT val, NTILE(2) OVER (ORDER BY id) AS bucket FROM items2")
        .expect("Failed to parse");

    let query = match stmt {
        SqlStatement::Select(q) => q.to_query(),
        _ => panic!("Expected SELECT"),
    };

    let planner = QueryPlanner::new();
    let plan = planner.plan(&query).expect("Failed to plan query");

    let executor = StorageExecutor::new(storage);
    let context = joule_db_query::QueryContext::new();
    let result = executor.execute(&plan, &context).unwrap();

    assert_eq!(result.rows.len(), 4);
    let bucket_idx = result
        .columns
        .iter()
        .position(|c| c == "bucket")
        .expect("bucket not found");
    // 4 rows / 2 buckets: first 2 in bucket 1, last 2 in bucket 2
    let buckets: Vec<_> = result.rows.iter().map(|r| &r.values[bucket_idx]).collect();
    assert_eq!(
        buckets,
        vec![
            &Value::Int(1),
            &Value::Int(1),
            &Value::Int(2),
            &Value::Int(2)
        ]
    );
}
