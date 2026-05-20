//! Amorphic Storage Executor
//!
//! Connects the SQL query executor to the AmorphicEngine storage,
//! enabling SQL queries over holographic/hyperdimensional data.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// Re-export for convenience
pub use crate::ast::{Expression, Value};
pub use crate::error::{QueryError, QueryResult};
pub use crate::executor::{RowData, TableStorage};

/// Schema definition for a table
#[derive(Debug, Clone)]
pub struct TableSchema {
    /// Table name
    pub name: String,
    /// Column names in order
    pub columns: Vec<String>,
    /// Primary key column (if any)
    pub primary_key: Option<String>,
}

impl TableSchema {
    /// Create a new table schema
    pub fn new(name: &str, columns: Vec<&str>) -> Self {
        Self {
            name: name.to_string(),
            columns: columns.into_iter().map(String::from).collect(),
            primary_key: None,
        }
    }

    /// Set primary key
    pub fn with_primary_key(mut self, column: &str) -> Self {
        self.primary_key = Some(column.to_string());
        self
    }
}

/// In-memory row storage
type TableData = HashMap<u64, RowData>;

/// Amorphic-backed table storage
///
/// Simple in-memory implementation that stores data in HashMaps.
/// This serves as the foundation for connecting SQL to AmorphicEngine.
pub struct AmorphicTableStorage {
    /// Table schemas
    schemas: RwLock<HashMap<String, TableSchema>>,
    /// Table data: table_name -> (row_id -> RowData)
    data: RwLock<HashMap<String, TableData>>,
    /// Row counter per table (for auto-increment)
    row_counters: RwLock<HashMap<String, u64>>,
}

impl AmorphicTableStorage {
    /// Create new amorphic table storage
    pub fn new() -> Self {
        Self {
            schemas: RwLock::new(HashMap::new()),
            data: RwLock::new(HashMap::new()),
            row_counters: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new table
    pub fn create_table(&self, schema: TableSchema) -> QueryResult<()> {
        let mut schemas = self
            .schemas
            .write()
            .expect("amorphic storage lock poisoned");
        let mut data = self.data.write().expect("amorphic storage lock poisoned");

        if schemas.contains_key(&schema.name) {
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' already exists",
                schema.name
            )));
        }

        let name = schema.name.clone();
        schemas.insert(name.clone(), schema);
        data.insert(name, HashMap::new());
        Ok(())
    }

    /// Drop a table
    pub fn drop_table(&self, name: &str) -> QueryResult<()> {
        let mut schemas = self
            .schemas
            .write()
            .expect("amorphic storage lock poisoned");
        let mut data = self.data.write().expect("amorphic storage lock poisoned");

        if schemas.remove(name).is_none() {
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' does not exist",
                name
            )));
        }
        data.remove(name);
        Ok(())
    }

    /// Get next row ID for a table
    fn next_row_id(&self, table: &str) -> u64 {
        let mut counters = self
            .row_counters
            .write()
            .expect("amorphic storage lock poisoned");
        let counter = counters.entry(table.to_string()).or_insert(0);
        *counter += 1;
        *counter
    }

    /// Evaluate a predicate against a row using canonical operator evaluation.
    ///
    /// Supports all comparison operators (Eq, Ne, Lt, Le, Gt, Ge),
    /// logical operators (And, Or, Not), and nested expressions.
    fn matches_predicate(row: &RowData, predicate: Option<&Expression>) -> bool {
        match predicate {
            None => true,
            Some(expr) => Self::eval_predicate_expr(row, expr),
        }
    }

    /// Recursively evaluate an expression against a row.
    fn eval_predicate_expr(row: &RowData, expr: &Expression) -> bool {
        match expr {
            Expression::Binary { left, op, right } => {
                let left_val = Self::resolve_expr_value(row, left);
                let right_val = Self::resolve_expr_value(row, right);
                match crate::functions::eval_binary_op(&left_val, op, &right_val) {
                    Ok(Value::Bool(b)) => b,
                    _ => false,
                }
            }
            Expression::Unary { op, expr: inner } => {
                let val = Self::resolve_expr_value(row, inner);
                match crate::functions::eval_unary_op(op, &val) {
                    Ok(Value::Bool(b)) => b,
                    _ => false,
                }
            }
            Expression::Literal(Value::Bool(b)) => *b,
            _ => true,
        }
    }

    /// Resolve an expression to a Value given a row context.
    fn resolve_expr_value(row: &RowData, expr: &Expression) -> Value {
        match expr {
            Expression::Column(col_name) => row.get(col_name).cloned().unwrap_or(Value::Null),
            Expression::Literal(val) => val.clone(),
            Expression::Binary { left, op, right } => {
                let l = Self::resolve_expr_value(row, left);
                let r = Self::resolve_expr_value(row, right);
                crate::functions::eval_binary_op(&l, op, &r).unwrap_or(Value::Null)
            }
            Expression::Unary { op, expr: inner } => {
                let v = Self::resolve_expr_value(row, inner);
                crate::functions::eval_unary_op(op, &v).unwrap_or(Value::Null)
            }
            _ => Value::Null,
        }
    }
}

impl Default for AmorphicTableStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl TableStorage for AmorphicTableStorage {
    fn scan(&self, table: &str) -> QueryResult<Vec<RowData>> {
        let schemas = self.schemas.read().expect("amorphic storage lock poisoned");
        if !schemas.contains_key(table) {
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' does not exist",
                table
            )));
        }
        drop(schemas);

        let data = self.data.read().expect("amorphic storage lock poisoned");
        let table_data = data
            .get(table)
            .ok_or_else(|| QueryError::ExecutionError(format!("Table '{}' has no data", table)))?;

        Ok(table_data.values().cloned().collect())
    }

    fn columns(&self, table: &str) -> QueryResult<Vec<String>> {
        let schemas = self.schemas.read().expect("amorphic storage lock poisoned");
        let schema = schemas.get(table).ok_or_else(|| {
            QueryError::ExecutionError(format!("Table '{}' does not exist", table))
        })?;
        Ok(schema.columns.clone())
    }

    fn insert(&self, table: &str, row: &RowData) -> QueryResult<()> {
        // Check table exists
        {
            let schemas = self.schemas.read().expect("amorphic storage lock poisoned");
            if !schemas.contains_key(table) {
                return Err(QueryError::ExecutionError(format!(
                    "Table '{}' does not exist",
                    table
                )));
            }
        }

        let row_id = self.next_row_id(table);
        let mut data = self.data.write().expect("amorphic storage lock poisoned");

        let table_data = data.entry(table.to_string()).or_insert_with(HashMap::new);
        table_data.insert(row_id, row.clone());

        Ok(())
    }

    fn update(
        &self,
        table: &str,
        assignments: &HashMap<String, Value>,
        predicate: Option<&Expression>,
    ) -> QueryResult<usize> {
        let mut data = self.data.write().expect("amorphic storage lock poisoned");
        let table_data = data.get_mut(table).ok_or_else(|| {
            QueryError::ExecutionError(format!("Table '{}' does not exist", table))
        })?;

        let mut updated = 0;
        for row in table_data.values_mut() {
            if Self::matches_predicate(row, predicate) {
                // Apply assignments
                for (col, val) in assignments {
                    if let Some(idx) = row.columns.iter().position(|c| c == col) {
                        row.values[idx] = val.clone();
                    }
                }
                updated += 1;
            }
        }

        Ok(updated)
    }

    fn delete(&self, table: &str, predicate: Option<&Expression>) -> QueryResult<usize> {
        let mut data = self.data.write().expect("amorphic storage lock poisoned");
        let table_data = data.get_mut(table).ok_or_else(|| {
            QueryError::ExecutionError(format!("Table '{}' does not exist", table))
        })?;

        let to_delete: Vec<u64> = table_data
            .iter()
            .filter(|(_, row)| Self::matches_predicate(row, predicate))
            .map(|(id, _)| *id)
            .collect();

        let count = to_delete.len();
        for id in to_delete {
            table_data.remove(&id);
        }

        Ok(count)
    }

    fn table_exists(&self, table: &str) -> QueryResult<bool> {
        let schemas = self.schemas.read().expect("amorphic storage lock poisoned");
        Ok(schemas.contains_key(table))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_table_and_insert() {
        let storage = AmorphicTableStorage::new();

        // Create table
        let schema = TableSchema::new("users", vec!["id", "name", "age"]);
        storage.create_table(schema).unwrap();

        // Insert row
        let row = RowData::new(
            vec!["id".to_string(), "name".to_string(), "age".to_string()],
            vec![
                Value::Int(1),
                Value::String("Alice".to_string()),
                Value::Int(30),
            ],
        );
        storage.insert("users", &row).unwrap();

        // Scan
        let rows = storage.scan("users").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].get("name"),
            Some(&Value::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_multiple_inserts_and_scan() {
        let storage = AmorphicTableStorage::new();

        let schema = TableSchema::new("products", vec!["id", "name", "price"]);
        storage.create_table(schema).unwrap();

        for i in 1..=5 {
            let row = RowData::new(
                vec!["id".to_string(), "name".to_string(), "price".to_string()],
                vec![
                    Value::Int(i),
                    Value::String(format!("Product {}", i)),
                    Value::Float(9.99 * i as f64),
                ],
            );
            storage.insert("products", &row).unwrap();
        }

        let rows = storage.scan("products").unwrap();
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn test_table_not_exists() {
        let storage = AmorphicTableStorage::new();

        let result = storage.scan("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_columns() {
        let storage = AmorphicTableStorage::new();

        let schema = TableSchema::new("test", vec!["a", "b", "c"]);
        storage.create_table(schema).unwrap();

        let cols = storage.columns("test").unwrap();
        assert_eq!(cols, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_delete_all() {
        let storage = AmorphicTableStorage::new();

        let schema = TableSchema::new("items", vec!["id"]);
        storage.create_table(schema).unwrap();

        for i in 1..=3 {
            let row = RowData::new(vec!["id".to_string()], vec![Value::Int(i)]);
            storage.insert("items", &row).unwrap();
        }

        let deleted = storage.delete("items", None).unwrap();
        assert_eq!(deleted, 3);

        let rows = storage.scan("items").unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_update() {
        let storage = AmorphicTableStorage::new();

        let schema = TableSchema::new("users", vec!["id", "name"]);
        storage.create_table(schema).unwrap();

        let row = RowData::new(
            vec!["id".to_string(), "name".to_string()],
            vec![Value::Int(1), Value::String("Alice".to_string())],
        );
        storage.insert("users", &row).unwrap();

        let mut assignments = HashMap::new();
        assignments.insert("name".to_string(), Value::String("Bob".to_string()));

        let updated = storage.update("users", &assignments, None).unwrap();
        assert_eq!(updated, 1);

        let rows = storage.scan("users").unwrap();
        assert_eq!(rows[0].get("name"), Some(&Value::String("Bob".to_string())));
    }

    #[test]
    fn test_comparison_operators_in_predicate() {
        let storage = AmorphicTableStorage::new();

        let schema = TableSchema::new("scores", vec!["id", "value"]);
        storage.create_table(schema).unwrap();

        for i in 1..=5 {
            let row = RowData::new(
                vec!["id".to_string(), "value".to_string()],
                vec![Value::Int(i), Value::Int(i * 10)],
            );
            storage.insert("scores", &row).unwrap();
        }

        // Test Gt: WHERE value > 30 should match rows with value 40, 50
        let predicate = Expression::Binary {
            left: Box::new(Expression::Column("value".to_string())),
            op: crate::ast::Operator::Gt,
            right: Box::new(Expression::Literal(Value::Int(30))),
        };
        let deleted = storage.delete("scores", Some(&predicate)).unwrap();
        assert_eq!(deleted, 2, "Gt predicate should match 2 rows (40, 50)");

        let remaining = storage.scan("scores").unwrap();
        assert_eq!(remaining.len(), 3);
    }

    #[test]
    fn test_similar_to_parsing() {
        use crate::sql::SqlParser;

        let mut parser = SqlParser::new();

        // Test SIMILAR TO with THRESHOLD
        let stmt = parser.parse("SELECT * FROM users WHERE name SIMILAR TO 'John' THRESHOLD 0.8");
        assert!(stmt.is_ok(), "Failed to parse SIMILAR TO: {:?}", stmt);

        // Test SIMILAR TO without THRESHOLD
        let stmt = parser.parse("SELECT * FROM users WHERE name SIMILAR TO 'Alice'");
        assert!(
            stmt.is_ok(),
            "Failed to parse SIMILAR TO without threshold: {:?}",
            stmt
        );
    }

    #[test]
    fn test_like_meaning_parsing() {
        use crate::sql::SqlParser;

        let mut parser = SqlParser::new();

        // Test LIKE MEANING
        let stmt = parser.parse("SELECT * FROM docs WHERE content LIKE MEANING 'machine learning'");
        assert!(stmt.is_ok(), "Failed to parse LIKE MEANING: {:?}", stmt);
    }
}
