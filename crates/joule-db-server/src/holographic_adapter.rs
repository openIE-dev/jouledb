use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use joule_db_hdc::holographic_kv::{HolographicKV, HolographicKVConfig};
use joule_db_query::ast::{Expression, Operator, Value};
use joule_db_query::error::{QueryError, QueryResult};
use joule_db_query::executor::{RowData, TableStorage};

/// Adapter to expose HolographicKV as a SQL-queryable table
///
/// This allows queries like `SELECT * FROM holographic`
/// or `INSERT INTO holographic (key, value) VALUES (...)`
pub struct HolographicTableStorage {
    // We use a fixed dimension of 512 for the standard adapter
    store: Arc<RwLock<HolographicKV<512>>>,
}

impl HolographicTableStorage {
    pub fn new() -> Self {
        let config = HolographicKVConfig::default();
        Self {
            store: Arc::new(RwLock::new(HolographicKV::new(config))),
        }
    }

    fn check_table(&self, table: &str) -> QueryResult<()> {
        if table != "holographic" && table != "kv" {
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' not found. This adapter only supports 'holographic' or 'kv'.",
                table
            )));
        }
        Ok(())
    }
}

impl TableStorage for HolographicTableStorage {
    fn scan(&self, table: &str) -> QueryResult<Vec<RowData>> {
        self.check_table(table)?;

        let store = self
            .store
            .read()
            .map_err(|_| QueryError::ExecutionError("Lock poisoned".to_string()))?;

        let mut rows = Vec::new();
        let columns = vec!["key".to_string(), "value".to_string()];

        for (key, value) in store.iter() {
            let row_values = vec![Value::Bytes(key.clone()), Value::Bytes(value.clone())];
            rows.push(RowData::new(columns.clone(), row_values));
        }

        Ok(rows)
    }

    fn columns(&self, table: &str) -> QueryResult<Vec<String>> {
        self.check_table(table)?;
        Ok(vec!["key".to_string(), "value".to_string()])
    }

    fn insert(&self, table: &str, row: &RowData) -> QueryResult<()> {
        self.check_table(table)?;

        let key = row
            .get("key")
            .ok_or_else(|| QueryError::ExecutionError("Missing 'key' column".to_string()))?;
        let value = row
            .get("value")
            .ok_or_else(|| QueryError::ExecutionError("Missing 'value' column".to_string()))?;

        let key_bytes = match key {
            Value::String(s) => s.as_bytes().to_vec(),
            Value::Bytes(b) => b.clone(),
            _ => {
                return Err(QueryError::ExecutionError(
                    "Key must be string or bytes".to_string(),
                ));
            }
        };

        let value_bytes = match value {
            Value::String(s) => s.as_bytes().to_vec(),
            Value::Bytes(b) => b.clone(),
            _ => {
                return Err(QueryError::ExecutionError(
                    "Value must be string or bytes".to_string(),
                ));
            }
        };

        let mut store = self
            .store
            .write()
            .map_err(|_| QueryError::ExecutionError("Lock poisoned".to_string()))?;

        store
            .put(&key_bytes, &value_bytes)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;

        Ok(())
    }

    fn update(
        &self,
        table: &str,
        assignments: &HashMap<String, Value>,
        _predicate: Option<&Expression>,
    ) -> QueryResult<usize> {
        self.check_table(table)?;
        // Update is complex because it requires read-modify-write with predicate filtering.
        // For simplicity in this adapter, we'll only support direct KV updates via INSERT (upsert) usually,
        // or minimal updates if keys are specified.
        // Properly implementing generic UPDATE with WHERE predicate on non-indexed field requires full scan.
        // For this MVP, we return error to encourage INSERT/DELETE usage.
        Err(QueryError::ExecutionError(
            "UPDATE not implemented for Holographic Store. Use INSERT (Upsert) instead."
                .to_string(),
        ))
    }

    fn delete(&self, table: &str, predicate: Option<&Expression>) -> QueryResult<usize> {
        self.check_table(table)?;

        // Only support DELETE WHERE key = ...
        // We can't easily support arbitrary complex predicates without scanning/evaluating first.
        // But for an adapter, let's try to handle specific simple "key = X" predicates if possible,
        // or just error for now.

        // This is a minimal implementation.
        Err(QueryError::ExecutionError("DELETE with predicate not implemented yet. Please assume append-only or use specific API.".to_string()))
    }

    fn table_exists(&self, table: &str) -> QueryResult<bool> {
        Ok(table == "holographic" || table == "kv")
    }

    fn index_scan(
        &self,
        table: &str,
        index: &str,
        predicate: &Expression,
    ) -> QueryResult<Vec<RowData>> {
        self.check_table(table)?;

        if index != "key" {
            return Err(QueryError::ExecutionError(format!(
                "Index '{}' not found. Only 'key' is indexed.",
                index
            )));
        }

        // Check if predicate is a simple equality check on 'key'
        let value = match predicate {
            Expression::Binary {
                left,
                op: Operator::Eq,
                right,
            } => {
                let left_col = match left.as_ref() {
                    Expression::Column(c) => Some(c.as_str()),
                    _ => None,
                };
                let right_col = match right.as_ref() {
                    Expression::Column(c) => Some(c.as_str()),
                    _ => None,
                };

                if left_col == Some("key") {
                    match right.as_ref() {
                        Expression::Literal(v) => Some(v),
                        _ => None,
                    }
                } else if right_col == Some("key") {
                    match left.as_ref() {
                        Expression::Literal(v) => Some(v),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(v) = value {
            let key_bytes = match v {
                Value::String(s) => s.as_bytes().to_vec(),
                Value::Bytes(b) => b.clone(),
                _ => return Ok(vec![]),
            };

            let store = self
                .store
                .read()
                .map_err(|_| QueryError::ExecutionError("Lock poisoned".to_string()))?;

            if let Some(val_bytes) = store.get(&key_bytes) {
                let row_values = vec![Value::Bytes(key_bytes), Value::Bytes(val_bytes)];
                let columns = vec!["key".to_string(), "value".to_string()];
                Ok(vec![RowData::new(columns, row_values)])
            } else {
                Ok(vec![])
            }
        } else {
            // Complex predicate or not matching index structure
            Err(QueryError::ExecutionError(
                "Index scan only supports 'key = literal' predicates".to_string(),
            ))
        }
    }
}
