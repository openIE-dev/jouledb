//! Schema management: create/drop tables, column definitions, column families, ALTER TABLE, sequences.

use super::*;
use super::conversions::*;

impl AmorphicTableStorage {
    /// Ensure a schema record exists for the given table, creating or merging as needed.
    pub(super) fn ensure_schema(
        &self,
        table: &str,
        store: &mut DurableAmorphicStore,
        record: &serde_json::Value,
    ) -> QueryResult<()> {
        let map = match record.as_object() {
            Some(m) => m,
            None => return Ok(()),
        };

        // Collect columns and types from the record (skip internal fields)
        let mut columns = Vec::new();
        let mut types = HashMap::new();
        for (key, value) in map {
            if key == TABLE_FIELD || key == COLLECTION_FIELD {
                continue;
            }
            columns.push(key.clone());
            types.insert(key.clone(), infer_column_type(value).to_string());
        }

        self.ensure_schema_with_columns(table, store, &columns, &types)
    }

    /// Core schema create/merge logic.
    pub(super) fn ensure_schema_with_columns(
        &self,
        table: &str,
        store: &mut DurableAmorphicStore,
        new_columns: &[String],
        new_types: &HashMap<String, String>,
    ) -> QueryResult<()> {
        // Check if schema already exists
        let existing = self.get_schema_columns_from_store(store, table);

        match existing {
            Ok(existing_cols) => {
                // Schema exists -- merge new columns
                let existing_set: std::collections::HashSet<&str> =
                    existing_cols.iter().map(|s| s.as_str()).collect();

                let mut merged_columns = existing_cols.clone();
                let mut has_new = false;

                for col in new_columns {
                    if !existing_set.contains(col.as_str()) {
                        merged_columns.push(col.clone());
                        has_new = true;
                    }
                }

                if !has_new {
                    return Ok(()); // No new columns, schema unchanged
                }

                // Delete old schema record and create updated one
                let result = store
                    .query_equals(SCHEMA_NAME_FIELD, &AmorphicValue::String(table.to_string()));
                for record in result.records() {
                    if record.get(TABLE_FIELD)
                        == Some(&AmorphicValue::String(SCHEMA_TABLE.to_string()))
                    {
                        let _ = store.delete(record.id);
                    }
                }

                // Create new schema with merged columns
                self.create_schema_record(store, table, &merged_columns, new_types)
            }
            Err(_) => {
                // Schema doesn't exist -- create it
                self.create_schema_record(store, table, new_columns, new_types)
            }
        }
    }

    /// Create a schema metadata record in the amorphic store.
    fn create_schema_record(
        &self,
        store: &mut DurableAmorphicStore,
        table: &str,
        columns: &[String],
        types: &HashMap<String, String>,
    ) -> QueryResult<()> {
        let columns_json: Vec<serde_json::Value> = columns
            .iter()
            .map(|c| serde_json::Value::String(c.clone()))
            .collect();

        let defs_json: Vec<serde_json::Value> = columns
            .iter()
            .map(|col| {
                let data_type = types.get(col).map(|s| s.as_str()).unwrap_or("TEXT");
                let obj = serde_json::json!({
                    "name": col,
                    "type": data_type,
                    "nullable": true,
                    "primary_key": false,
                    "default": null,
                });
                serde_json::Value::String(obj.to_string())
            })
            .collect();

        let schema_json = serde_json::json!({
            TABLE_FIELD: SCHEMA_TABLE,
            SCHEMA_NAME_FIELD: table,
            SCHEMA_COLUMNS_FIELD: columns_json,
            SCHEMA_COLUMN_DEFS_FIELD: defs_json,
        });

        store
            .ingest_json(&schema_json.to_string())
            .map_err(amorphic_error)?;

        Ok(())
    }

    /// Extract schema columns using a borrowed store reference (avoids re-locking)
    pub(super) fn get_schema_columns_from_store(
        &self,
        store: &DurableAmorphicStore,
        table: &str,
    ) -> QueryResult<Vec<String>> {
        self.get_schema_columns(store, table)
    }

    /// Create a SQL table (stores schema metadata as an amorphic record)
    pub fn create_table(&self, name: &str, columns: &[String]) -> QueryResult<()> {
        if name.starts_with("__") {
            return Err(QueryError::ExecutionError(format!(
                "Table name '{}' is reserved (starts with __)",
                name
            )));
        }

        // Check if table already exists
        if self.table_exists(name)? {
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' already exists",
                name
            )));
        }

        // Store schema as an amorphic record
        let columns_json: Vec<serde_json::Value> = columns
            .iter()
            .map(|c| serde_json::Value::String(c.clone()))
            .collect();

        let schema_json = serde_json::json!({
            TABLE_FIELD: SCHEMA_TABLE,
            SCHEMA_NAME_FIELD: name,
            SCHEMA_COLUMNS_FIELD: columns_json,
        });

        let mut store = self.store.write().map_err(lock_error)?;
        store
            .ingest_json(&schema_json.to_string())
            .map_err(amorphic_error)?;

        Ok(())
    }

    /// Create a SQL table with full column definitions
    pub fn create_table_with_defs(
        &self,
        name: &str,
        columns: &[String],
        column_defs: &[SqlColumnDef],
    ) -> QueryResult<()> {
        if name.starts_with("__") {
            return Err(QueryError::ExecutionError(format!(
                "Table name '{}' is reserved (starts with __)",
                name
            )));
        }

        if self.table_exists(name)? {
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' already exists",
                name
            )));
        }

        let columns_json: Vec<serde_json::Value> = columns
            .iter()
            .map(|c| serde_json::Value::String(c.clone()))
            .collect();

        let defs_json: Vec<serde_json::Value> = column_defs
            .iter()
            .map(|def| {
                // Serialize each def as a JSON string to avoid AmorphicValue::Object complexity
                let default_val = def.default.as_ref().map(|e| expr_to_default_value(e));
                let check_str = def.check.as_ref().map(|e| expression_to_sql(e));
                let fk_json = def.foreign_key.as_ref().map(|fk| {
                    use joule_db_query::sql::ReferentialAction;
                    let action_str = |a: &ReferentialAction| -> &str {
                        match a {
                            ReferentialAction::Cascade => "CASCADE",
                            ReferentialAction::Restrict => "RESTRICT",
                            ReferentialAction::SetNull => "SET NULL",
                            ReferentialAction::NoAction => "NO ACTION",
                        }
                    };
                    serde_json::json!({
                        "ref_table": fk.ref_table,
                        "ref_column": fk.ref_column,
                        "on_delete": fk.on_delete.as_ref().map(|a| action_str(a)),
                        "on_update": fk.on_update.as_ref().map(|a| action_str(a)),
                    })
                });
                let computed_str = def.computed.as_ref().map(|e| expression_to_sql(e));
                let obj = serde_json::json!({
                    "name": def.name,
                    "type": def.data_type,
                    "nullable": def.nullable,
                    "primary_key": def.primary_key,
                    "unique": def.unique,
                    "default": default_val,
                    "check_expr": check_str,
                    "auto_increment": def.auto_increment,
                    "foreign_key": fk_json,
                    "column_family": def.column_family,
                    "computed_expr": computed_str,
                });
                serde_json::Value::String(obj.to_string())
            })
            .collect();

        let schema_json = serde_json::json!({
            TABLE_FIELD: SCHEMA_TABLE,
            SCHEMA_NAME_FIELD: name,
            SCHEMA_COLUMNS_FIELD: columns_json,
            SCHEMA_COLUMN_DEFS_FIELD: defs_json,
        });

        let mut store = self.store.write().map_err(lock_error)?;
        store
            .ingest_json(&schema_json.to_string())
            .map_err(amorphic_error)?;

        Ok(())
    }

    /// Get full column definitions for a table (returns empty vec if not stored)
    pub fn get_column_defs(&self, table: &str) -> QueryResult<Vec<ColumnDefInfo>> {
        let store = self.store.read().map_err(lock_error)?;
        self.get_column_defs_from_store(&store, table)
    }

    pub(super) fn get_column_defs_from_store(
        &self,
        store: &DurableAmorphicStore,
        table: &str,
    ) -> QueryResult<Vec<ColumnDefInfo>> {
        let result =
            store.query_equals(SCHEMA_NAME_FIELD, &AmorphicValue::String(table.to_string()));
        for record in result.records() {
            if record.get(TABLE_FIELD) == Some(&AmorphicValue::String(SCHEMA_TABLE.to_string())) {
                if let Some(AmorphicValue::Array(defs)) = record.get(SCHEMA_COLUMN_DEFS_FIELD) {
                    return Ok(defs
                        .iter()
                        .filter_map(|v| {
                            if let AmorphicValue::String(json_str) = v {
                                serde_json::from_str::<serde_json::Value>(json_str)
                                    .ok()
                                    .map(|obj| {
                                        let default_value = obj.get("default").and_then(|v| {
                                            if v.is_null() { None } else { Some(v.clone()) }
                                        });
                                        let foreign_key = obj.get("foreign_key").and_then(|fk| {
                                            if fk.is_null() {
                                                return None;
                                            }
                                            Some(ForeignKeyInfo {
                                                ref_table: fk
                                                    .get("ref_table")?
                                                    .as_str()?
                                                    .to_string(),
                                                ref_column: fk
                                                    .get("ref_column")?
                                                    .as_str()?
                                                    .to_string(),
                                                on_delete: fk
                                                    .get("on_delete")
                                                    .and_then(|v| v.as_str())
                                                    .map(|s| s.to_string()),
                                                on_update: fk
                                                    .get("on_update")
                                                    .and_then(|v| v.as_str())
                                                    .map(|s| s.to_string()),
                                            })
                                        });
                                        ColumnDefInfo {
                                            name: obj
                                                .get("name")
                                                .and_then(|v| v.as_str())
                                                .unwrap_or("")
                                                .to_string(),
                                            data_type: obj
                                                .get("type")
                                                .and_then(|v| v.as_str())
                                                .unwrap_or("TEXT")
                                                .to_string(),
                                            nullable: obj
                                                .get("nullable")
                                                .and_then(|v| v.as_bool())
                                                .unwrap_or(true),
                                            primary_key: obj
                                                .get("primary_key")
                                                .and_then(|v| v.as_bool())
                                                .unwrap_or(false),
                                            unique: obj
                                                .get("unique")
                                                .and_then(|v| v.as_bool())
                                                .unwrap_or(false),
                                            default_value,
                                            check_expr: obj
                                                .get("check_expr")
                                                .and_then(|v| v.as_str())
                                                .map(|s| s.to_string()),
                                            auto_increment: obj
                                                .get("auto_increment")
                                                .and_then(|v| v.as_bool())
                                                .unwrap_or(false),
                                            foreign_key,
                                            column_family: obj
                                                .get("column_family")
                                                .and_then(|v| v.as_str())
                                                .map(|s| s.to_string()),
                                            computed_expr: obj
                                                .get("computed_expr")
                                                .and_then(|v| v.as_str())
                                                .map(|s| s.to_string()),
                                        }
                                    })
                            } else {
                                None
                            }
                        })
                        .collect());
                }
            }
        }
        // No column defs stored (backward compat) -- return empty vec
        Ok(Vec::new())
    }

    /// Set column families for a table (wide-column storage).
    /// Stores the family names in the schema record.
    pub fn set_column_families(&self, table: &str, families: &[String]) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;
        // Find the schema record for this table
        let result =
            store.query_equals(SCHEMA_NAME_FIELD, &AmorphicValue::String(table.to_string()));
        let schema_record = result
            .records()
            .iter()
            .find(|r| r.get(TABLE_FIELD) == Some(&AmorphicValue::String(SCHEMA_TABLE.to_string())))
            .ok_or_else(|| QueryError::ExecutionError(format!("Table '{}' not found", table)))?;

        let record_id = schema_record.id;

        // Build the update fields
        let families_json: Vec<serde_json::Value> = families
            .iter()
            .map(|f| serde_json::Value::String(f.clone()))
            .collect();
        let mut updates = std::collections::HashMap::new();
        updates.insert(
            SCHEMA_COLUMN_FAMILIES_FIELD.to_string(),
            AmorphicValue::String(serde_json::to_string(&families_json).map_err(|e| {
                QueryError::ExecutionError(format!("Failed to serialize column families: {}", e))
            })?),
        );

        store
            .update_fields(record_id, updates)
            .map_err(amorphic_error)?;
        Ok(())
    }

    /// Get column families for a table (returns empty vec if not defined).
    pub fn get_column_families(&self, table: &str) -> QueryResult<Vec<String>> {
        let store = self.store.read().map_err(lock_error)?;
        let result =
            store.query_equals(SCHEMA_NAME_FIELD, &AmorphicValue::String(table.to_string()));
        for record in result.records() {
            if record.get(TABLE_FIELD) == Some(&AmorphicValue::String(SCHEMA_TABLE.to_string())) {
                if let Some(AmorphicValue::String(json_str)) =
                    record.get(SCHEMA_COLUMN_FAMILIES_FIELD)
                {
                    if let Ok(families) = serde_json::from_str::<Vec<String>>(json_str) {
                        return Ok(families);
                    }
                }
            }
        }
        Ok(Vec::new())
    }

    /// Drop a SQL table (removes schema record and all data records)
    pub fn drop_table(&self, name: &str) -> QueryResult<bool> {
        let mut store = self.store.write().map_err(lock_error)?;

        // Find and delete the schema record
        let schema_result =
            store.query_equals(SCHEMA_NAME_FIELD, &AmorphicValue::String(name.to_string()));
        let schema_records: Vec<RecordId> = schema_result
            .records()
            .iter()
            .filter(|r| {
                r.get(TABLE_FIELD) == Some(&AmorphicValue::String(SCHEMA_TABLE.to_string()))
            })
            .map(|r| r.id)
            .collect();

        if schema_records.is_empty() {
            return Ok(false);
        }

        for id in &schema_records {
            let _ = store.delete(*id);
        }

        // Delete all data records belonging to this table
        let data_result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(name.to_string()));
        let data_ids: Vec<RecordId> = data_result.records().iter().map(|r| r.id).collect();
        for id in data_ids {
            let _ = store.delete(id);
        }

        Ok(true)
    }

    // ==================== ALTER TABLE operations ====================

    /// Add a column to an existing table's schema.
    /// Existing rows implicitly have NULL for the new column (schema-on-read).
    pub fn alter_add_column(&self, table: &str, column: &str) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;
        let mut columns = self.get_schema_columns(&store, table)?;

        if columns.iter().any(|c| c == column) {
            return Err(QueryError::ExecutionError(format!(
                "Column '{}' already exists in table '{}'",
                column, table
            )));
        }

        columns.push(column.to_string());
        self.update_schema_columns(&mut store, table, &columns)
    }

    /// Drop a column from an existing table's schema.
    /// Blocks if any index references the column.
    pub fn alter_drop_column(&self, table: &str, column: &str) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        if !columns.iter().any(|c| c == column) {
            return Err(QueryError::ExecutionError(format!(
                "Column '{}' does not exist in table '{}'",
                column, table
            )));
        }

        if columns.len() == 1 {
            return Err(QueryError::ExecutionError(
                "Cannot drop the only column in a table".to_string(),
            ));
        }

        // Check that no index references this column
        drop(store);
        let indexes = self.list_indexes(table)?;
        for (idx_name, idx_cols, _, _) in &indexes {
            if idx_cols.iter().any(|c| c == column) {
                return Err(QueryError::ExecutionError(format!(
                    "Cannot drop column '{}': referenced by index '{}'",
                    column, idx_name
                )));
            }
        }

        let mut store = self.store.write().map_err(lock_error)?;
        let new_columns: Vec<String> = columns.into_iter().filter(|c| c != column).collect();
        self.update_schema_columns(&mut store, table, &new_columns)
    }

    /// Rename a column in a table's schema and update all data rows.
    pub fn alter_rename_column(
        &self,
        table: &str,
        old_name: &str,
        new_name: &str,
    ) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        if !columns.iter().any(|c| c == old_name) {
            return Err(QueryError::ExecutionError(format!(
                "Column '{}' does not exist in table '{}'",
                old_name, table
            )));
        }

        if columns.iter().any(|c| c == new_name) {
            return Err(QueryError::ExecutionError(format!(
                "Column '{}' already exists in table '{}'",
                new_name, table
            )));
        }

        // Update schema
        let new_columns: Vec<String> = columns
            .iter()
            .map(|c| {
                if c == old_name {
                    new_name.to_string()
                } else {
                    c.clone()
                }
            })
            .collect();
        self.update_schema_columns(&mut store, table, &new_columns)?;

        // Update data rows: find all records for this table, rename the field
        let data_result =
            store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let records: Vec<(RecordId, AmorphicRecord)> = data_result
            .records()
            .iter()
            .map(|r| (r.id, r.clone()))
            .collect();

        for (id, record) in records {
            if let Some(_val) = record.get(old_name) {
                let _ = store.delete(id);

                // Re-ingest with renamed field
                let mut fields = std::collections::HashMap::new();
                fields.insert(
                    TABLE_FIELD.to_string(),
                    AmorphicValue::String(table.to_string()),
                );
                for col in &new_columns {
                    let field_name = if col == new_name {
                        old_name
                    } else {
                        col.as_str()
                    };
                    if let Some(v) = record.get(field_name) {
                        fields.insert(col.clone(), v.clone());
                    }
                }
                // Re-ingest as JSON
                let json_obj: serde_json::Map<String, serde_json::Value> = fields
                    .into_iter()
                    .map(|(k, v)| (k, amorphic_value_to_json(&v)))
                    .collect();
                let json_str = serde_json::Value::Object(json_obj).to_string();
                store.ingest_json(&json_str).map_err(amorphic_error)?;
            }
        }

        // Update index metadata if any index references the old column name
        drop(store);
        let indexes = self.list_indexes(table)?;
        for (idx_name, idx_cols, idx_unique, _) in &indexes {
            if idx_cols.iter().any(|c| c == old_name) {
                let new_idx_cols: Vec<String> = idx_cols
                    .iter()
                    .map(|c| {
                        if c == old_name {
                            new_name.to_string()
                        } else {
                            c.clone()
                        }
                    })
                    .collect();
                // Drop and re-create with updated columns
                self.drop_index(idx_name)?;
                self.create_index(idx_name, table, &new_idx_cols, *idx_unique, false)?;
            }
        }

        Ok(())
    }

    /// Helper: replace the schema columns for a table (delete old schema record, insert new).
    /// Preserves any `__column_defs__` from the old record.
    pub(super) fn update_schema_columns(
        &self,
        store: &mut DurableAmorphicStore,
        table: &str,
        columns: &[String],
    ) -> QueryResult<()> {
        // Find old schema record and extract column_defs before deleting
        let result =
            store.query_equals(SCHEMA_NAME_FIELD, &AmorphicValue::String(table.to_string()));
        let mut old_column_defs: Option<AmorphicValue> = None;
        let schema_ids: Vec<RecordId> = result
            .records()
            .iter()
            .filter(|r| {
                r.get(TABLE_FIELD) == Some(&AmorphicValue::String(SCHEMA_TABLE.to_string()))
            })
            .map(|r| {
                if old_column_defs.is_none() {
                    old_column_defs = r.get(SCHEMA_COLUMN_DEFS_FIELD).cloned();
                }
                r.id
            })
            .collect();
        for id in schema_ids {
            let _ = store.delete(id);
        }

        // Insert new schema record, preserving column_defs if they existed
        let columns_json: Vec<serde_json::Value> = columns
            .iter()
            .map(|c| serde_json::Value::String(c.clone()))
            .collect();
        let mut schema_map = serde_json::Map::new();
        schema_map.insert(TABLE_FIELD.to_string(), serde_json::json!(SCHEMA_TABLE));
        schema_map.insert(SCHEMA_NAME_FIELD.to_string(), serde_json::json!(table));
        schema_map.insert(
            SCHEMA_COLUMNS_FIELD.to_string(),
            serde_json::json!(columns_json),
        );

        if let Some(defs) = old_column_defs {
            // Convert AmorphicValue back to serde_json for re-ingestion
            schema_map.insert(
                SCHEMA_COLUMN_DEFS_FIELD.to_string(),
                amorphic_to_json(&defs),
            );
        }

        let schema_json = serde_json::Value::Object(schema_map).to_string();
        store.ingest_json(&schema_json).map_err(amorphic_error)?;
        Ok(())
    }

    /// Get the next auto-increment value for a table.column sequence.
    pub fn next_sequence_value(&self, table: &str, column: &str) -> QueryResult<i64> {
        let seq_name = format!("{}.{}", table, column);
        let mut store = self.store.write().map_err(lock_error)?;

        // Find existing sequence record
        let result = store.query_equals(
            SEQUENCE_NAME_FIELD,
            &AmorphicValue::String(seq_name.clone()),
        );
        let records = result.records();

        let (current_val, old_record_id) = if let Some(record) = records.first() {
            let val: i64 = record
                .get(SEQUENCE_VALUE_FIELD)
                .and_then(|v| {
                    if let AmorphicValue::Int(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            (val, Some(record.id))
        } else {
            (0, None)
        };

        let next_val = current_val + 1;

        // Delete old sequence record if exists
        if let Some(rid) = old_record_id {
            store.delete(rid).map_err(amorphic_error)?;
        }

        // Insert updated sequence value
        let seq_json = serde_json::json!({
            TABLE_FIELD: SEQUENCE_TABLE,
            SEQUENCE_NAME_FIELD: seq_name,
            SEQUENCE_VALUE_FIELD: next_val,
        });
        store
            .ingest_json(&seq_json.to_string())
            .map_err(amorphic_error)?;

        Ok(next_val)
    }
}
