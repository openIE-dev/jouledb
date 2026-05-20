//! CRUD operations: ingest, get, delete, query, batch operations, scan methods, RBAC persistence.

use super::*;
use super::conversions::*;

impl AmorphicTableStorage {
    // ==================== Direct amorphic API (for REST endpoints) ====================

    /// Ingest a raw JSON document into the amorphic store (no __table__ convention)
    pub fn ingest_json(&self, json: &str) -> Result<u64, joule_db_amorphic::AmorphicError> {
        let mut store = self.store.write().map_err(|_| {
            joule_db_amorphic::AmorphicError::IngestionError("Lock poisoned".into())
        })?;
        store.ingest_json(json)
    }

    /// Ingest a JSON document with automatic schema inference.
    pub fn ingest_with_schema(
        &self,
        json: &str,
        collection: Option<&str>,
    ) -> QueryResult<(u64, String)> {
        let mut obj: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| QueryError::ExecutionError(format!("Invalid JSON: {}", e)))?;

        let map = obj
            .as_object_mut()
            .ok_or_else(|| QueryError::ExecutionError("Expected JSON object".into()))?;

        // Determine collection name: explicit param > _collection field > __default__
        let coll = if let Some(c) = collection {
            c.to_string()
        } else if let Some(serde_json::Value::String(c)) = map.remove(COLLECTION_FIELD) {
            c
        } else {
            DEFAULT_COLLECTION.to_string()
        };

        // Set __table__ for SQL queryability
        map.insert(
            TABLE_FIELD.to_string(),
            serde_json::Value::String(coll.clone()),
        );

        let mut store = self.store.write().map_err(lock_error)?;

        // Ensure schema exists or merge new columns
        self.ensure_schema(&coll, &mut store, &obj)?;

        // Ingest the record
        let id = store
            .ingest_json(&obj.to_string())
            .map_err(amorphic_error)?;

        Ok((id, coll))
    }

    /// Batch ingest JSON documents with automatic schema inference.
    pub fn batch_ingest_with_schema(
        &self,
        records: &[serde_json::Value],
        collection: Option<&str>,
    ) -> QueryResult<(Vec<u64>, String)> {
        if records.is_empty() {
            let coll = collection.unwrap_or(DEFAULT_COLLECTION).to_string();
            return Ok((vec![], coll));
        }

        let mut store = self.store.write().map_err(lock_error)?;

        // Determine collection name from first record or param
        let coll = if let Some(c) = collection {
            c.to_string()
        } else if let Some(serde_json::Value::String(c)) =
            records[0].as_object().and_then(|m| m.get(COLLECTION_FIELD))
        {
            c.clone()
        } else {
            DEFAULT_COLLECTION.to_string()
        };

        // Build merged schema from all records
        let mut all_columns: Vec<String> = Vec::new();
        let mut all_types: HashMap<String, String> = HashMap::new();

        for record in records {
            if let Some(map) = record.as_object() {
                for (key, value) in map {
                    if key == COLLECTION_FIELD || key == TABLE_FIELD {
                        continue;
                    }
                    if !all_types.contains_key(key) {
                        all_columns.push(key.clone());
                        all_types.insert(key.clone(), infer_column_type(value).to_string());
                    }
                }
            }
        }

        // Ensure schema
        self.ensure_schema_with_columns(&coll, &mut store, &all_columns, &all_types)?;

        // Ingest all records
        let mut ids = Vec::with_capacity(records.len());
        for record in records {
            let mut obj = record.clone();
            if let Some(map) = obj.as_object_mut() {
                map.remove(COLLECTION_FIELD);
                map.insert(
                    TABLE_FIELD.to_string(),
                    serde_json::Value::String(coll.clone()),
                );
            }
            let id = store
                .ingest_json(&obj.to_string())
                .map_err(amorphic_error)?;
            ids.push(id);
        }

        Ok((ids, coll))
    }

    /// Ingest a graph edge
    pub fn ingest_edge(
        &self,
        source: &str,
        relation: &str,
        target: &str,
    ) -> Result<u64, joule_db_amorphic::AmorphicError> {
        let mut store = self.store.write().map_err(|_| {
            joule_db_amorphic::AmorphicError::IngestionError("Lock poisoned".into())
        })?;
        store.ingest_edge(source, relation, target)
    }

    /// Get a record by ID as a JSON document
    pub fn get_record(&self, id: u64) -> Option<serde_json::Value> {
        let store = self.store.read().ok()?;
        let record = store.get(id)?;
        let mut map = serde_json::Map::new();
        map.insert("id".to_string(), serde_json::json!(record.id));
        for field_name in record.field_names() {
            if let Some(val) = record.get(field_name) {
                map.insert(field_name.to_string(), amorphic_value_to_json(val));
            }
        }
        Some(serde_json::Value::Object(map))
    }

    /// Delete a record by ID
    pub fn delete_record(&self, id: u64) -> Result<(), joule_db_amorphic::AmorphicError> {
        let mut store = self.store.write().map_err(|_| {
            joule_db_amorphic::AmorphicError::IngestionError("Lock poisoned".into())
        })?;
        store.delete(id)
    }

    /// Query similar records by name
    pub fn query_similar_to(&self, name: &str, k: usize) -> Vec<serde_json::Value> {
        let store = match self.store.read() {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let result = store.query_similar_to(name, k);
        result
            .as_documents()
            .into_iter()
            .map(|doc| {
                let map: serde_json::Map<String, serde_json::Value> = doc
                    .into_iter()
                    .map(|(k, v)| (k, amorphic_value_to_json(&v)))
                    .collect();
                serde_json::Value::Object(map)
            })
            .collect()
    }

    /// Query graph traversal
    pub fn query_graph(&self, start: &str, relation: &str, depth: usize) -> Vec<serde_json::Value> {
        let store = match self.store.read() {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let result = store.query_graph(start, relation, depth);
        result
            .as_documents()
            .into_iter()
            .map(|doc| {
                let map: serde_json::Map<String, serde_json::Value> = doc
                    .into_iter()
                    .map(|(k, v)| (k, amorphic_value_to_json(&v)))
                    .collect();
                serde_json::Value::Object(map)
            })
            .collect()
    }

    // ==================== Batch transaction API (for MVCC commit) ====================

    /// Begin a batch transaction on the underlying DurableAmorphicStore.
    pub fn begin_batch(&self) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;
        store.begin_batch().map_err(amorphic_error)?;
        Ok(())
    }

    /// Insert a SQL row into the amorphic store within the current batch transaction.
    pub fn batch_insert(&self, table: &str, row: &RowData) -> QueryResult<RecordId> {
        let mut store = self.store.write().map_err(lock_error)?;

        // Build JSON with __table__ field
        let mut fields = serde_json::Map::new();
        fields.insert(
            TABLE_FIELD.to_string(),
            serde_json::Value::String(table.to_string()),
        );
        for (col, val) in row.columns.iter().zip(row.values.iter()) {
            fields.insert(col.clone(), ast_to_json(val));
        }

        let json = serde_json::Value::Object(fields).to_string();
        let id = store.batch_ingest_json(&json).map_err(amorphic_error)?;
        Ok(id)
    }

    /// Delete a record by amorphic RecordId within the current batch transaction.
    pub fn batch_delete_record(&self, id: RecordId) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;
        store.batch_delete(id).map_err(amorphic_error)?;
        Ok(())
    }

    /// Update fields of a record by amorphic RecordId within the current batch.
    pub fn batch_update_record(
        &self,
        id: RecordId,
        updates: HashMap<String, AmorphicValue>,
    ) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;
        store
            .batch_update_fields(id, updates)
            .map_err(amorphic_error)?;
        Ok(())
    }

    /// Commit the current batch transaction (atomic WAL commit + fsync).
    pub fn commit_batch(&self) -> QueryResult<usize> {
        let mut store = self.store.write().map_err(lock_error)?;
        let count = store.commit_batch().map_err(amorphic_error)?;
        Ok(count)
    }

    /// Rollback the current batch transaction.
    pub fn rollback_batch(&self) -> QueryResult<()> {
        let mut store = self.store.write().map_err(lock_error)?;
        store.rollback_batch().map_err(amorphic_error)?;
        Ok(())
    }

    /// Check if the WAL needs a checkpoint and perform one if so.
    pub fn checkpoint_if_needed(&self) -> QueryResult<bool> {
        let mut store = self.store.write().map_err(lock_error)?;
        if store.needs_checkpoint() {
            store.checkpoint().map_err(amorphic_error)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Scan all rows for a table, returning each row paired with its amorphic RecordId.
    pub fn scan_with_ids(&self, table: &str) -> QueryResult<Vec<(RecordId, RowData)>> {
        let store = self.store.read().map_err(lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let rows = result
            .records()
            .iter()
            .map(|r| (r.id, Self::record_to_row(r, &columns)))
            .collect();

        Ok(rows)
    }

    /// Find the amorphic RecordId for a record matching the given table, columns, and values.
    pub fn find_record_id(
        &self,
        table: &str,
        columns: &[String],
        values: &[AmorphicValue],
    ) -> Option<RecordId> {
        let store = self.store.read().ok()?;
        store.find_record_id(table, columns, values)
    }

    /// Ingest a JSON document into a specific table (sets __table__ field).
    pub fn ingest_with_table(&self, json: &str, table: &str) -> QueryResult<u64> {
        let mut obj: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| QueryError::ExecutionError(format!("Invalid JSON: {}", e)))?;
        let map = obj
            .as_object_mut()
            .ok_or_else(|| QueryError::ExecutionError("Expected JSON object".into()))?;
        map.insert(
            TABLE_FIELD.to_string(),
            serde_json::Value::String(table.to_string()),
        );
        let mut store = self.store.write().map_err(lock_error)?;
        store.ingest_json(&obj.to_string()).map_err(amorphic_error)
    }

    /// Delete all records in a table (used for materialized view refresh).
    pub fn delete_all(&self, table: &str) -> QueryResult<usize> {
        let mut store = self.store.write().map_err(lock_error)?;
        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let ids: Vec<RecordId> = result.records().iter().map(|r| r.id).collect();
        let count = ids.len();
        for id in ids {
            let _ = store.delete(id);
        }
        Ok(count)
    }

    // ============ Additional scan methods ============

    /// Check if a table exists in the schema (public API for HRP Phase 2 delta apply).
    pub fn has_table(&self, table: &str) -> bool {
        let store = match self.store.read() {
            Ok(s) => s,
            Err(_) => return false,
        };
        let result =
            store.query_equals(SCHEMA_NAME_FIELD, &AmorphicValue::String(table.to_string()));
        result
            .records()
            .iter()
            .any(|r| r.get(TABLE_FIELD) == Some(&AmorphicValue::String(SCHEMA_TABLE.to_string())))
    }

    /// Scan table rows, returning each row paired with its amorphic record ID.
    pub fn scan_with_record_ids(&self, table: &str) -> QueryResult<Vec<(String, RowData)>> {
        let store = self.store.read().map_err(lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let rows = result
            .records()
            .iter()
            .map(|r| {
                let record_id = r.id.to_string();
                let row = Self::record_to_row(r, &columns);
                (record_id, row)
            })
            .collect();

        Ok(rows)
    }

    /// Scan a table's column for f32 vector data, returning (row_id, vector) pairs.
    pub fn scan_vectors_for_index(
        &self,
        table: &str,
        column: &str,
    ) -> QueryResult<Vec<(String, Vec<f32>)>> {
        let rows = self.scan_with_record_ids(table)?;
        let mut result = Vec::new();
        for (record_id, row) in &rows {
            if let Some(val) = row.get(column) {
                // Try to extract a vector from the value
                let vec_f32 = match val {
                    joule_db_query::ast::Value::Array(arr) => {
                        let v: Vec<f32> = arr
                            .iter()
                            .filter_map(|v| match v {
                                joule_db_query::ast::Value::Float(f) => Some(*f as f32),
                                joule_db_query::ast::Value::Int(i) => Some(*i as f32),
                                _ => None,
                            })
                            .collect();
                        if v.len() == arr.len() { Some(v) } else { None }
                    }
                    joule_db_query::ast::Value::String(s) => {
                        // Try parsing "[1.0, 2.0, 3.0]" format
                        let trimmed = s.trim().trim_start_matches('[').trim_end_matches(']');
                        let parsed: Vec<f32> = trimmed
                            .split(',')
                            .filter_map(|p| p.trim().parse::<f32>().ok())
                            .collect();
                        if !parsed.is_empty() {
                            Some(parsed)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some(v) = vec_f32 {
                    result.push((record_id.clone(), v));
                }
            }
        }
        Ok(result)
    }

    /// Scan a table's column for spatial (x, y) data, returning (row_id, x, y) triples.
    pub fn scan_spatial_for_index(
        &self,
        table: &str,
        column: &str,
    ) -> QueryResult<Vec<(String, f64, f64)>> {
        let rows = self.scan_with_record_ids(table)?;
        let mut result = Vec::new();
        for (record_id, row) in &rows {
            if let Some(val) = row.get(column) {
                // Try to extract (x, y) from array or string
                let coords = match val {
                    joule_db_query::ast::Value::Array(arr) if arr.len() >= 2 => {
                        let x = match &arr[0] {
                            joule_db_query::ast::Value::Float(f) => Some(*f),
                            joule_db_query::ast::Value::Int(i) => Some(*i as f64),
                            _ => None,
                        };
                        let y = match &arr[1] {
                            joule_db_query::ast::Value::Float(f) => Some(*f),
                            joule_db_query::ast::Value::Int(i) => Some(*i as f64),
                            _ => None,
                        };
                        x.zip(y)
                    }
                    _ => None,
                };
                if let Some((x, y)) = coords {
                    result.push((record_id.clone(), x, y));
                }
            }
        }
        Ok(result)
    }

    /// Insert a row and return the assigned amorphic record ID.
    pub fn insert_returning_id(&self, table: &str, row: &RowData) -> QueryResult<u64> {
        let mut store = self.store.write().map_err(lock_error)?;

        // Verify table exists
        let _columns = self.get_schema_columns(&store, table)?;

        // Build JSON with __table__ field
        let mut fields = serde_json::Map::new();
        fields.insert(
            TABLE_FIELD.to_string(),
            serde_json::Value::String(table.to_string()),
        );

        for (col, val) in row.columns.iter().zip(row.values.iter()) {
            fields.insert(col.clone(), ast_to_json(val));
        }

        let json = serde_json::Value::Object(fields).to_string();
        let record_id = store.ingest_json(&json).map_err(amorphic_error)?;

        Ok(record_id)
    }

    // ============ RBAC Persistence Methods ============

    /// Scan RBAC meta-table records. Returns JSON objects from raw amorphic store.
    pub fn scan_rbac_meta(&self, meta_table: &str) -> Vec<serde_json::Value> {
        let store = match self.store.read() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let result =
            store.query_equals(TABLE_FIELD, &AmorphicValue::String(meta_table.to_string()));
        result
            .records()
            .iter()
            .map(|r| {
                let mut map = serde_json::Map::new();
                map.insert("__record_id__".to_string(), serde_json::json!(r.id));
                for field_name in r.field_names() {
                    if let Some(v) = r.get(field_name) {
                        match v {
                            AmorphicValue::String(s) => {
                                map.insert(field_name.to_string(), serde_json::json!(s));
                            }
                            AmorphicValue::Int(n) => {
                                map.insert(field_name.to_string(), serde_json::json!(n));
                            }
                            AmorphicValue::Float(f) => {
                                map.insert(field_name.to_string(), serde_json::json!(f));
                            }
                            AmorphicValue::Bool(b) => {
                                map.insert(field_name.to_string(), serde_json::json!(b));
                            }
                            AmorphicValue::Null => {
                                map.insert(field_name.to_string(), serde_json::Value::Null);
                            }
                            _ => {
                                map.insert(
                                    field_name.to_string(),
                                    serde_json::json!(format!("{:?}", v)),
                                );
                            }
                        }
                    }
                }
                serde_json::Value::Object(map)
            })
            .collect()
    }

    /// Insert a JSON record into a RBAC meta-table.
    pub fn insert_rbac_meta(&self, json: &serde_json::Value) {
        if let Ok(mut store) = self.store.write() {
            let _ = store.ingest_json(&json.to_string());
        }
    }

    /// Delete all records from a RBAC meta-table matching a field value.
    pub fn delete_rbac_meta_by_field(&self, meta_table: &str, field: &str, value: &str) {
        if let Ok(store) = self.store.read() {
            let result =
                store.query_equals(TABLE_FIELD, &AmorphicValue::String(meta_table.to_string()));
            let ids: Vec<u64> = result
                .records()
                .iter()
                .filter(|r| matches!(r.get(field), Some(AmorphicValue::String(s)) if s == value))
                .map(|r| r.id)
                .collect();
            drop(store);
            if let Ok(mut store) = self.store.write() {
                for id in ids {
                    let _ = store.delete(id);
                }
            }
        }
    }
}
