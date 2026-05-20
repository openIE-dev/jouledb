use joule_db_query::{
    Expression, QueryPlanner, QueryResult, RowData, SqlParser, SqlStatement, StorageExecutor,
    TableStorage, Value,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// Mock Storage
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

#[test]
fn test_simple_cte() {
    let storage = Arc::new(MemoryTableStorage::new());
    storage.create_table(
        "users",
        vec!["id".to_string(), "name".to_string(), "age".to_string()],
    );
    storage.insert_row(
        "users",
        vec![
            Value::Int(1),
            Value::String("Alice".to_string()),
            Value::Int(30),
        ],
    );
    storage.insert_row(
        "users",
        vec![
            Value::Int(2),
            Value::String("Bob".to_string()),
            Value::Int(25),
        ],
    );

    let mut parser = SqlParser::new();
    let stmt = parser
        .parse(
            "
        WITH older_users AS (
            SELECT * FROM users WHERE age > 28
        )
        SELECT name FROM older_users
    ",
        )
        .unwrap();

    let query = match stmt {
        SqlStatement::Select(q) => q.to_query(),
        _ => panic!("Expected SELECT"),
    };

    let planner = QueryPlanner::new();
    let plan = planner.plan(&query).expect("Failed to plan query");

    let executor = StorageExecutor::new(storage);
    let context = joule_db_query::QueryContext::new();
    let result = executor
        .execute(&plan, &context)
        .expect("Failed to execute query");

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].values[0], Value::String("Alice".to_string()));
}

#[test]
fn test_cte_chaining() {
    let storage = Arc::new(MemoryTableStorage::new());
    storage.create_table(
        "users",
        vec!["id".to_string(), "name".to_string(), "age".to_string()],
    );
    storage.insert_row(
        "users",
        vec![
            Value::Int(1),
            Value::String("Alice".to_string()),
            Value::Int(30),
        ],
    );

    // Note: This relies on recursive CTE lookup in planner
    let mut parser = SqlParser::new();
    let stmt = parser
        .parse(
            "
        WITH t1 AS (SELECT * FROM users),
             t2 AS (SELECT * FROM t1 WHERE age > 20)
        SELECT name FROM t2
    ",
        )
        .unwrap();

    let query = match stmt {
        SqlStatement::Select(q) => q.to_query(),
        _ => panic!("Expected SELECT"),
    };

    let planner = QueryPlanner::new();
    let plan = planner
        .plan(&query)
        .expect("Failed to plan chained CTE query");

    let executor = StorageExecutor::new(storage);
    let result = executor
        .execute(&plan, &joule_db_query::QueryContext::new())
        .expect("Execution failed");

    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].values[0], Value::String("Alice".to_string()));
}
