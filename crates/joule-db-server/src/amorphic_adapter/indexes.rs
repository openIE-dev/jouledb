//! Index management: B-tree, spatial, and vector indexes; table statistics.

use super::*;
use super::conversions::*;

impl AmorphicTableStorage {
    /// Create an index on a table's columns.
    pub fn create_index(
        &self,
        name: &str,
        table: &str,
        columns: &[String],
        unique: bool,
        if_not_exists: bool,
    ) -> QueryResult<()> {
        // Verify table exists
        if !self.table_exists(table)? {
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' does not exist",
                table
            )));
        }

        // Check if index already exists
        let store = self.store.read().map_err(lock_error)?;
        let existing =
            store.query_equals(INDEX_NAME_FIELD, &AmorphicValue::String(name.to_string()));
        let exists = existing
            .records()
            .iter()
            .any(|r| r.get(TABLE_FIELD) == Some(&AmorphicValue::String(INDEX_TABLE.to_string())));
        drop(store);

        if exists {
            if if_not_exists {
                return Ok(());
            }
            return Err(QueryError::ExecutionError(format!(
                "Index '{}' already exists",
                name
            )));
        }

        // Store index metadata
        let columns_json: Vec<serde_json::Value> = columns
            .iter()
            .map(|c| serde_json::Value::String(c.clone()))
            .collect();

        let index_json = serde_json::json!({
            TABLE_FIELD: INDEX_TABLE,
            INDEX_NAME_FIELD: name,
            INDEX_TABLE_REF: table,
            INDEX_COLUMNS_FIELD: columns_json,
            INDEX_UNIQUE_FIELD: unique,
        });

        let mut store = self.store.write().map_err(lock_error)?;
        store
            .ingest_json(&index_json.to_string())
            .map_err(amorphic_error)?;

        Ok(())
    }

    /// Drop an index by name. Returns true if it existed.
    pub fn drop_index(&self, name: &str) -> QueryResult<bool> {
        let mut store = self.store.write().map_err(lock_error)?;
        let result = store.query_equals(INDEX_NAME_FIELD, &AmorphicValue::String(name.to_string()));
        let index_ids: Vec<RecordId> = result
            .records()
            .iter()
            .filter(|r| r.get(TABLE_FIELD) == Some(&AmorphicValue::String(INDEX_TABLE.to_string())))
            .map(|r| r.id)
            .collect();

        if index_ids.is_empty() {
            return Ok(false);
        }

        for id in &index_ids {
            let _ = store.delete(*id);
        }
        Ok(true)
    }

    /// List all indexes for a table, returning (name, columns, unique, index_type) tuples.
    pub fn list_indexes(
        &self,
        table: &str,
    ) -> QueryResult<Vec<(String, Vec<String>, bool, String)>> {
        let store = self.store.read().map_err(lock_error)?;
        let result = store.query_equals(INDEX_TABLE_REF, &AmorphicValue::String(table.to_string()));
        let mut indexes = Vec::new();

        for record in result.records() {
            if record.get(TABLE_FIELD) != Some(&AmorphicValue::String(INDEX_TABLE.to_string())) {
                continue;
            }
            let name = match record.get(INDEX_NAME_FIELD) {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => continue,
            };
            let columns = match record.get(INDEX_COLUMNS_FIELD) {
                Some(AmorphicValue::Array(arr)) => arr
                    .iter()
                    .filter_map(|v| match v {
                        AmorphicValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect(),
                _ => continue,
            };
            let unique = match record.get(INDEX_UNIQUE_FIELD) {
                Some(AmorphicValue::Bool(b)) => *b,
                _ => false,
            };
            let index_type = match record.get("__index_type__") {
                Some(AmorphicValue::String(s)) => s.clone(),
                _ => "btree".to_string(),
            };
            indexes.push((name, columns, unique, index_type));
        }

        Ok(indexes)
    }

    // ==================== Statistics ====================

    /// Collect real statistics for a table (row count, column cardinality, min/max).
    pub fn get_table_statistics(&self, table: &str) -> QueryResult<TableStatistics> {
        let store = self.store.read().map_err(lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let records = result.records();
        let row_count = records.len();

        let mut col_stats = HashMap::new();
        for col_name in &columns {
            let mut distinct: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut null_count = 0usize;
            let mut min_num = f64::MAX;
            let mut max_num = f64::MIN;
            let mut has_numeric = false;

            for record in records {
                match record.get(col_name) {
                    Some(AmorphicValue::Null) | None => null_count += 1,
                    Some(val) => {
                        distinct.insert(format!("{:?}", val));
                        match val {
                            AmorphicValue::Int(i) => {
                                let f = *i as f64;
                                min_num = min_num.min(f);
                                max_num = max_num.max(f);
                                has_numeric = true;
                            }
                            AmorphicValue::Float(f) => {
                                min_num = min_num.min(*f);
                                max_num = max_num.max(*f);
                                has_numeric = true;
                            }
                            _ => {}
                        }
                    }
                }
            }

            col_stats.insert(
                col_name.clone(),
                ColumnStatistics {
                    distinct_count: distinct.len(),
                    null_fraction: if row_count > 0 {
                        null_count as f64 / row_count as f64
                    } else {
                        0.0
                    },
                    min_value: if has_numeric {
                        Some(min_num.to_string())
                    } else {
                        None
                    },
                    max_value: if has_numeric {
                        Some(max_num.to_string())
                    } else {
                        None
                    },
                },
            );
        }

        drop(store);
        let indexes = self.list_indexes(table).unwrap_or_default();

        Ok(TableStatistics {
            row_count,
            columns: col_stats,
            indexes,
        })
    }

    /// Collect planner-compatible statistics for all known tables.
    pub fn collect_planner_statistics(&self) -> PlannerStatistics {
        let mut stats = PlannerStatistics::new();

        // List all tables by scanning schema records
        let store = match self.store.read() {
            Ok(s) => s,
            Err(_) => return stats,
        };

        let schema_result = store.query_equals(
            TABLE_FIELD,
            &AmorphicValue::String(SCHEMA_TABLE.to_string()),
        );

        let mut table_names = Vec::new();
        for record in schema_result.records() {
            if let Some(AmorphicValue::String(name)) = record.get(SCHEMA_NAME_FIELD) {
                table_names.push(name.clone());
            }
        }
        drop(store);

        for table_name in &table_names {
            if let Ok(ts) = self.get_table_statistics(table_name) {
                let mut planner_cols = HashMap::new();
                for (col_name, col_stat) in &ts.columns {
                    planner_cols.insert(
                        col_name.clone(),
                        PlannerColumnStats {
                            distinct_count: col_stat.distinct_count,
                            null_fraction: col_stat.null_fraction,
                            min_value: col_stat.min_value.clone(),
                            max_value: col_stat.max_value.clone(),
                            most_common: Vec::new(),
                            histogram: Vec::new(),
                        },
                    );
                }

                let mut planner_indexes = Vec::new();
                for (idx_name, idx_cols, idx_unique, idx_type_str) in &ts.indexes {
                    let index_type = match idx_type_str.as_str() {
                        "fulltext" => IndexType::FullText,
                        "spatial" => IndexType::Spatial,
                        "vector" => IndexType::Vector,
                        _ => IndexType::BTree,
                    };
                    planner_indexes.push(IndexInfo {
                        name: idx_name.clone(),
                        columns: idx_cols.clone(),
                        unique: *idx_unique,
                        index_type,
                    });
                }

                stats.add_table(
                    table_name,
                    PlannerTableStats {
                        row_count: ts.row_count,
                        columns: planner_cols,
                        indexes: planner_indexes,
                    },
                );
            }
        }

        stats
    }

    // ================================================================
    // Spatial Index Management
    // ================================================================

    /// Create a spatial index on a geometry column.
    pub fn create_spatial_index(&self, name: &str, table: &str, column: &str) -> QueryResult<()> {
        // Verify table exists
        if !self.table_exists(table).unwrap_or(false) {
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' does not exist",
                table
            )));
        }

        // Check for duplicate index name
        let store = self.store.read().map_err(lock_error)?;
        let existing =
            store.query_equals(INDEX_NAME_FIELD, &AmorphicValue::String(name.to_string()));
        for record in existing.records() {
            if record.get(TABLE_FIELD) == Some(&AmorphicValue::String(INDEX_TABLE.to_string())) {
                return Err(QueryError::ExecutionError(format!(
                    "Index '{}' already exists",
                    name
                )));
            }
        }
        drop(store);

        // Store index metadata
        let index_json = serde_json::json!({
            TABLE_FIELD: INDEX_TABLE,
            INDEX_NAME_FIELD: name,
            INDEX_TABLE_REF: table,
            INDEX_COLUMNS_FIELD: [column],
            "__index_type__": "spatial",
        });

        let mut store = self.store.write().map_err(lock_error)?;
        store
            .ingest_json(&index_json.to_string())
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;

        Ok(())
    }

    /// Drop a spatial index by name.
    pub fn drop_spatial_index(&self, name: &str) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;
        let result = store.query_equals(INDEX_NAME_FIELD, &AmorphicValue::String(name.to_string()));
        let index_ids: Vec<RecordId> = result
            .records()
            .iter()
            .filter(|r| {
                r.get(TABLE_FIELD) == Some(&AmorphicValue::String(INDEX_TABLE.to_string()))
                    && r.get("__index_type__")
                        == Some(&AmorphicValue::String("spatial".to_string()))
            })
            .map(|r| r.id)
            .collect();

        if index_ids.is_empty() {
            return Err(QueryError::ExecutionError(format!(
                "Spatial index '{}' does not exist",
                name
            )));
        }

        for id in &index_ids {
            let _ = store.delete(*id);
        }

        Ok(())
    }

    // ================================================================
    // Vector Index Management
    // ================================================================

    /// Create a vector index on an embedding column.
    pub fn create_vector_index(
        &self,
        name: &str,
        table: &str,
        column: &str,
        method: &str,
        options: &std::collections::HashMap<String, String>,
    ) -> QueryResult<()> {
        // Verify table exists
        if !self.table_exists(table).unwrap_or(false) {
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' does not exist",
                table
            )));
        }

        // Check for duplicate index name
        let store = self.store.read().map_err(lock_error)?;
        let existing =
            store.query_equals(INDEX_NAME_FIELD, &AmorphicValue::String(name.to_string()));
        for record in existing.records() {
            if record.get(TABLE_FIELD) == Some(&AmorphicValue::String(INDEX_TABLE.to_string())) {
                return Err(QueryError::ExecutionError(format!(
                    "Index '{}' already exists",
                    name
                )));
            }
        }
        drop(store);

        // Store index metadata
        let options_json = serde_json::to_string(options).unwrap_or_default();
        let index_json = serde_json::json!({
            TABLE_FIELD: INDEX_TABLE,
            INDEX_NAME_FIELD: name,
            INDEX_TABLE_REF: table,
            INDEX_COLUMNS_FIELD: [column],
            "__index_type__": "vector",
            "__vector_method__": method,
            "__vector_options__": options_json,
        });

        let mut store = self.store.write().map_err(lock_error)?;
        store
            .ingest_json(&index_json.to_string())
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;

        Ok(())
    }

    /// Drop a vector index by name.
    pub fn drop_vector_index(&self, name: &str) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;
        let result = store.query_equals(INDEX_NAME_FIELD, &AmorphicValue::String(name.to_string()));
        let index_ids: Vec<RecordId> = result
            .records()
            .iter()
            .filter(|r| {
                r.get(TABLE_FIELD) == Some(&AmorphicValue::String(INDEX_TABLE.to_string()))
                    && r.get("__index_type__") == Some(&AmorphicValue::String("vector".to_string()))
            })
            .map(|r| r.id)
            .collect();

        if index_ids.is_empty() {
            return Err(QueryError::ExecutionError(format!(
                "Vector index '{}' does not exist",
                name
            )));
        }

        for id in &index_ids {
            let _ = store.delete(*id);
        }

        Ok(())
    }
}
