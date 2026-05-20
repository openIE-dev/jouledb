//! View and materialized view management.

use super::*;
use super::conversions::*;

impl AmorphicTableStorage {
    /// Create or replace a view.
    pub fn create_view(
        &self,
        name: &str,
        query_sql: &str,
        columns: Option<&[String]>,
        or_replace: bool,
    ) -> QueryResult<()> {
        // Check if view already exists
        let store = self.store.read().map_err(lock_error)?;
        let existing: Vec<_> = store
            .query_equals("__view_name__", &AmorphicValue::String(name.to_string()))
            .records()
            .iter()
            .map(|r| r.id)
            .collect();
        drop(store);

        if !existing.is_empty() {
            if or_replace {
                let mut store = self.store.write().map_err(lock_error)?;
                for id in existing {
                    store.delete(id).map_err(amorphic_error)?;
                }
            } else {
                return Err(QueryError::ExecutionError(format!(
                    "View '{}' already exists",
                    name
                )));
            }
        }

        let view_json = if let Some(cols) = columns {
            serde_json::json!({
                TABLE_FIELD: "__views__",
                "__view_name__": name,
                "__view_query__": query_sql,
                "__view_columns__": serde_json::to_string(cols).unwrap_or_default(),
            })
        } else {
            serde_json::json!({
                TABLE_FIELD: "__views__",
                "__view_name__": name,
                "__view_query__": query_sql,
            })
        };

        let mut store = self.store.write().map_err(lock_error)?;
        store
            .ingest_json(&view_json.to_string())
            .map_err(amorphic_error)?;
        Ok(())
    }

    /// Drop a view by name.
    pub fn drop_view(&self, name: &str) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;

        let ids: Vec<_> = store
            .query_equals("__view_name__", &AmorphicValue::String(name.to_string()))
            .records()
            .iter()
            .map(|r| r.id)
            .collect();

        if ids.is_empty() {
            return Err(QueryError::ExecutionError(format!(
                "View '{}' does not exist",
                name
            )));
        }

        for id in ids {
            store.delete(id).map_err(amorphic_error)?;
        }

        Ok(())
    }

    /// Get a view's query SQL and optional column aliases.
    pub fn get_view(&self, name: &str) -> Option<(String, Option<Vec<String>>)> {
        let store = self.store.read().ok()?;
        let result = store.query_equals("__view_name__", &AmorphicValue::String(name.to_string()));
        let records = result.records();
        let record = records.first()?;

        let query = match record.get("__view_query__") {
            Some(AmorphicValue::String(s)) => s.clone(),
            _ => return None,
        };

        let columns = record.get("__view_columns__").and_then(|v| match v {
            AmorphicValue::String(s) => serde_json::from_str::<Vec<String>>(s).ok(),
            _ => None,
        });

        Some((query, columns))
    }

    /// List all view names.
    pub fn list_views(&self) -> Vec<String> {
        let store = match self.store.read() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let result =
            store.query_equals(TABLE_FIELD, &AmorphicValue::String("__views__".to_string()));
        result
            .records()
            .iter()
            .filter_map(|r| match r.get("__view_name__") {
                Some(AmorphicValue::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect()
    }

    // ============ Materialized view management ============

    /// Create a materialized view definition (stores metadata, not data).
    pub fn create_materialized_view(
        &self,
        name: &str,
        query_sql: &str,
        backing_table: &str,
    ) -> QueryResult<()> {
        let json = serde_json::json!({
            TABLE_FIELD: MATVIEW_TABLE,
            MATVIEW_NAME_FIELD: name,
            MATVIEW_QUERY_FIELD: query_sql,
            MATVIEW_BACKING_TABLE_FIELD: backing_table,
        });
        let mut store = self.store.write().map_err(lock_error)?;
        store
            .ingest_json(&json.to_string())
            .map_err(amorphic_error)?;
        Ok(())
    }

    /// Get a materialized view definition: (query_sql, backing_table)
    pub fn get_materialized_view(&self, name: &str) -> QueryResult<Option<(String, String)>> {
        let store = self.store.read().map_err(lock_error)?;
        let result =
            store.query_equals(MATVIEW_NAME_FIELD, &AmorphicValue::String(name.to_string()));
        for record in result.records() {
            if record.get(TABLE_FIELD) == Some(&AmorphicValue::String(MATVIEW_TABLE.to_string())) {
                let query = match record.get(MATVIEW_QUERY_FIELD) {
                    Some(AmorphicValue::String(s)) => s.clone(),
                    _ => continue,
                };
                let backing = match record.get(MATVIEW_BACKING_TABLE_FIELD) {
                    Some(AmorphicValue::String(s)) => s.clone(),
                    _ => continue,
                };
                return Ok(Some((query, backing)));
            }
        }
        Ok(None)
    }

    /// Drop a materialized view definition. Returns true if it existed.
    pub fn drop_materialized_view(&self, name: &str) -> QueryResult<bool> {
        let mut store = self.store.write().map_err(lock_error)?;
        let result =
            store.query_equals(MATVIEW_NAME_FIELD, &AmorphicValue::String(name.to_string()));
        let ids: Vec<RecordId> = result
            .records()
            .iter()
            .filter(|r| {
                r.get(TABLE_FIELD) == Some(&AmorphicValue::String(MATVIEW_TABLE.to_string()))
            })
            .map(|r| r.id)
            .collect();
        if ids.is_empty() {
            return Ok(false);
        }
        for id in ids {
            let _ = store.delete(id);
        }
        Ok(true)
    }

    /// List all materialized view names.
    pub fn list_materialized_views(&self) -> Vec<String> {
        let store = match self.store.read() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let result = store.query_equals(
            TABLE_FIELD,
            &AmorphicValue::String(MATVIEW_TABLE.to_string()),
        );
        result
            .records()
            .iter()
            .filter_map(|r| match r.get(MATVIEW_NAME_FIELD) {
                Some(AmorphicValue::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect()
    }
}
