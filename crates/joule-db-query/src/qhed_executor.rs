//! QHED Storage Executor
//!
//! Connects the SQL query executor to the QHED storage engine,
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

/// QHED-backed table storage
///
/// Simple in-memory implementation that stores data in HashMaps.
/// This serves as the foundation for connecting SQL to QHED.
pub struct QHEDTableStorage {
    /// Table schemas
    schemas: RwLock<HashMap<String, TableSchema>>,
    /// Table data: table_name -> (row_id -> RowData)
    data: RwLock<HashMap<String, TableData>>,
    /// Row counter per table (for auto-increment)
    row_counters: RwLock<HashMap<String, u64>>,
}

impl QHEDTableStorage {
    /// Create new QHED table storage
    pub fn new() -> Self {
        Self {
            schemas: RwLock::new(HashMap::new()),
            data: RwLock::new(HashMap::new()),
            row_counters: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new table
    pub fn create_table(&self, schema: TableSchema) -> QueryResult<()> {
        let mut schemas = self.schemas.write().unwrap();
        let mut data = self.data.write().unwrap();

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
        let mut schemas = self.schemas.write().unwrap();
        let mut data = self.data.write().unwrap();

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
        let mut counters = self.row_counters.write().unwrap();
        let counter = counters.entry(table.to_string()).or_insert(0);
        *counter += 1;
        *counter
    }

    /// Evaluate a predicate against a row (simplified)
    fn matches_predicate(row: &RowData, predicate: Option<&Expression>) -> bool {
        match predicate {
            None => true,
            Some(Expression::Binary { left, op, right }) => {
                // Handle simple column = value comparisons
                if let (Expression::Column(col_name), Expression::Literal(val)) =
                    (left.as_ref(), right.as_ref())
                {
                    if let Some(row_val) = row.get(col_name) {
                        return match op {
                            crate::ast::Operator::Eq => Self::values_equal(row_val, val),
                            crate::ast::Operator::Ne => !Self::values_equal(row_val, val),
                            _ => true, // Default to true for unsupported operators
                        };
                    }
                }
                true
            }
            _ => true, // Default to true for complex predicates
        }
    }

    /// Compare two values for equality
    fn values_equal(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => x == y,
            (Value::Float(x), Value::Float(y)) => (x - y).abs() < f64::EPSILON,
            (Value::String(x), Value::String(y)) => x == y,
            (Value::Bool(x), Value::Bool(y)) => x == y,
            (Value::Null, Value::Null) => true,
            _ => false,
        }
    }
}

impl Default for QHEDTableStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl TableStorage for QHEDTableStorage {
    fn scan(&self, table: &str) -> QueryResult<Vec<RowData>> {
        let schemas = self.schemas.read().unwrap();
        if !schemas.contains_key(table) {
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' does not exist",
                table
            )));
        }
        drop(schemas);

        let data = self.data.read().unwrap();
        let table_data = data
            .get(table)
            .ok_or_else(|| QueryError::ExecutionError(format!("Table '{}' has no data", table)))?;

        Ok(table_data.values().cloned().collect())
    }

    fn columns(&self, table: &str) -> QueryResult<Vec<String>> {
        let schemas = self.schemas.read().unwrap();
        let schema = schemas.get(table).ok_or_else(|| {
            QueryError::ExecutionError(format!("Table '{}' does not exist", table))
        })?;
        Ok(schema.columns.clone())
    }

    fn insert(&self, table: &str, row: &RowData) -> QueryResult<()> {
        // Check table exists
        {
            let schemas = self.schemas.read().unwrap();
            if !schemas.contains_key(table) {
                return Err(QueryError::ExecutionError(format!(
                    "Table '{}' does not exist",
                    table
                )));
            }
        }

        let row_id = self.next_row_id(table);
        let mut data = self.data.write().unwrap();

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
        let mut data = self.data.write().unwrap();
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
        let mut data = self.data.write().unwrap();
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
        let schemas = self.schemas.read().unwrap();
        Ok(schemas.contains_key(table))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_table_and_insert() {
        let storage = QHEDTableStorage::new();

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
        let storage = QHEDTableStorage::new();

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
        let storage = QHEDTableStorage::new();

        let result = storage.scan("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_columns() {
        let storage = QHEDTableStorage::new();

        let schema = TableSchema::new("test", vec!["a", "b", "c"]);
        storage.create_table(schema).unwrap();

        let cols = storage.columns("test").unwrap();
        assert_eq!(cols, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_delete_all() {
        let storage = QHEDTableStorage::new();

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
        let storage = QHEDTableStorage::new();

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
