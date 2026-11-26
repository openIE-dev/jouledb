//! CRUD operations: ingest, get, delete, query, batch operations, scan methods, RBAC persistence.

use super::conversions::*;
use super::*;

/// Single operation inside an atomic MVCC-to-WAL commit batch.
pub enum AtomicBatchOp {
    Insert {
        key_idx: usize,
        table: String,
        row: RowData,
    },
    Update {
        id: RecordId,
        updates: HashMap<String, AmorphicValue>,
    },
    Delete {
        id: RecordId,
    },
}

/// Group-commit durability waiter returned after releasing the store write lock.
pub struct DeferredBatchSync {
    pub(crate) sync: Option<std::sync::Arc<joule_db_local::storage::wal::GroupSync>>,
    pub(crate) lsn: Option<u64>,
}

impl DeferredBatchSync {
    /// Wait for the batch's WAL LSN to become power-loss durable (shared fsync).
    pub fn wait_durable(self) -> QueryResult<()> {
        if let (Some(g), Some(lsn)) = (self.sync, self.lsn) {
            if let Err(e) = g.wait_durable(lsn) {
                tracing::error!("commit durability: {e}");
                SYNC_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(QueryError::ExecutionError(format!(
                    "commit not durable: {e}"
                )));
            }
        }
        Ok(())
    }
}

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
        let store = match self.store.read_indexed() {
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

    /// Execute an entire batch of inserts, updates, and deletes atomically under a
    /// single exclusive store lock, releasing the lock before returning a
    /// [`DeferredBatchSync`] handle so concurrent committers coalesce into a single
    /// `GroupSync::wait_durable` fsync.
    pub fn apply_atomic_batch_deferred(
        &self,
        ops: Vec<AtomicBatchOp>,
    ) -> QueryResult<(Vec<(usize, RecordId)>, DeferredBatchSync)> {
        let (mut guard, sync_arc) = self.store.write_deferred_sync().map_err(lock_error)?;
        guard.begin_batch().map_err(amorphic_error)?;

        let mut assigned_ids = Vec::new();
        for op in ops {
            let step_res: QueryResult<()> = match op {
                AtomicBatchOp::Insert {
                    key_idx,
                    table,
                    row,
                } => {
                    let mut fields = serde_json::Map::new();
                    fields.insert(
                        TABLE_FIELD.to_string(),
                        serde_json::Value::String(table),
                    );
                    for (col, val) in row.columns.iter().zip(row.values.iter()) {
                        fields.insert(col.clone(), ast_to_json(val));
                    }
                    let json = serde_json::Value::Object(fields).to_string();
                    match guard.batch_ingest_json(&json) {
                        Ok(id) => {
                            assigned_ids.push((key_idx, id));
                            Ok(())
                        }
                        Err(e) => Err(amorphic_error(e)),
                    }
                }
                AtomicBatchOp::Update { id, updates } => {
                    guard.batch_update_fields(id, updates).map_err(amorphic_error)
                }
                AtomicBatchOp::Delete { id } => guard.batch_delete(id).map_err(amorphic_error),
            };
            if let Err(e) = step_res {
                let _ = guard.rollback_batch();
                return Err(e);
            }
        }

        if let Err(e) = guard.commit_batch() {
            let _ = guard.rollback_batch();
            return Err(amorphic_error(e));
        }

        let lsn = sync_arc.as_ref().map(|g| g.committed_lsn());
        drop(guard);
        Ok((
            assigned_ids,
            DeferredBatchSync {
                sync: sync_arc,
                lsn,
            },
        ))
    }

    /// Look up rows in `table` matching `col == val` using `AmorphicStore`'s secondary
    /// field index in O(k), returning `None` if `col` is a floating-point column or `val`
    /// is `Float`/`Null`/`Array`/`Object` (so cross-type numeric equality falls back safely).
    pub fn lookup_records_by_eq(
        &self,
        table: &str,
        col: &str,
        val: &AstValue,
    ) -> QueryResult<Option<Vec<(RecordId, RowData)>>> {
        if !matches!(
            val,
            AstValue::Int(_) | AstValue::String(_) | AstValue::Bool(_)
        ) {
            return Ok(None);
        }
        let store = self.store.read().map_err(lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;
        if !columns.iter().any(|c| c == col) {
            return Ok(None);
        }
        if let Ok(defs) = self.get_column_defs_from_store(&store, table) {
            if let Some(def) = defs.iter().find(|d| d.name == col) {
                let dt = def.data_type.to_uppercase();
                let type_ok = match val {
                    AstValue::Int(_) => matches!(
                        dt.as_str(),
                        "INT" | "INTEGER" | "BIGINT" | "SMALLINT" | "SERIAL" | "BIGSERIAL"
                    ),
                    AstValue::String(_) => {
                        matches!(dt.as_str(), "TEXT" | "VARCHAR" | "CHAR" | "STRING" | "UUID")
                    }
                    AstValue::Bool(_) => matches!(dt.as_str(), "BOOL" | "BOOLEAN"),
                    _ => false,
                };
                if !type_ok && !dt.is_empty() {
                    return Ok(None);
                }
            } else if matches!(val, AstValue::Int(_)) {
                // Untyped schema: avoid missing Float(42.0) == Int(42)
                return Ok(None);
            }
        }
        let amorphic_val = ast_to_amorphic(val);
        let table_tag = AmorphicValue::String(table.to_string());
        let rows = store
            .records_equal(col, &amorphic_val)
            .into_iter()
            .filter(|r| r.get(TABLE_FIELD) == Some(&table_tag))
            .map(|r| (r.id, Self::record_to_row(r, &columns)))
            .collect();
        Ok(Some(rows))
    }

    /// Fetch a single row of `table` by its amorphic `RecordId`.
    pub fn get_table_row_by_id(&self, table: &str, id: RecordId) -> QueryResult<Option<RowData>> {
        let store = self.store.read().map_err(lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;
        let Some(rec) = store.get(id) else {
            return Ok(None);
        };
        if rec.get(TABLE_FIELD) != Some(&AmorphicValue::String(table.to_string())) {
            return Ok(None);
        }
        Ok(Some(Self::record_to_row(rec, &columns)))
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

        let rows = store
            .records_equal(TABLE_FIELD, &AmorphicValue::String(table.to_string()))
            .into_iter()
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
        let mut count = ids.len();
        if !ids.is_empty() {
            let _ = store.delete_batch(&ids);
        }
        drop(store);
        if let Ok(mut c) = self.col_cache.lock() {
            c.remove(table);
        }
        if let Ok(mut dc) = self.disk_cols.write()
            && let Some(dt) = dc.remove(table)
        {
            count += dt.total_rows;
            let _ = crate::disk_col_segments::DiskColumnTable::remove_table_files(
                &self.db_path,
                table,
            );
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
    /// Insert many rows in one WAL group commit (bulk COPY / multi-row
    /// INSERT). Same records as `insert_returning_id` per row, built
    /// directly from AST values (no JSON round trip).
    pub fn insert_rows(&self, table: &str, rows: &[RowData]) -> QueryResult<Vec<u64>> {
        use rayon::prelude::*;
        let t_prep = std::time::Instant::now();
        let batch: Vec<HashMap<String, AmorphicValue>> = rows
            .par_iter()
            .map(|row| {
                let mut f = HashMap::with_capacity(row.columns.len() + 1);
                f.insert(
                    TABLE_FIELD.to_string(),
                    AmorphicValue::String(table.to_string()),
                );
                for (c, v) in row.columns.iter().zip(row.values.iter()) {
                    f.insert(c.clone(), ast_into_amorphic(v.clone()));
                }
                f
            })
            .collect();
        self.insert_field_maps(table, batch, t_prep.elapsed())
    }

    /// `insert_rows` taking the rows by value: cells are moved into the
    /// record maps (no clone, no JSON round trip). The maps are built before
    /// the store write lock is taken, so readers and the previous batch's
    /// apply are not held up by the conversion.
    pub fn insert_rows_owned(&self, table: &str, rows: Vec<RowData>) -> QueryResult<Vec<u64>> {
        let t_prep = std::time::Instant::now();
        let batch = Self::rows_to_field_maps(table, rows);
        self.insert_field_maps(table, batch, t_prep.elapsed())
    }

    /// Owned rows -> record field maps (parallel; `_table` tag added).
    pub fn rows_to_field_maps(
        table: &str,
        rows: Vec<RowData>,
    ) -> Vec<HashMap<String, AmorphicValue>> {
        use rayon::prelude::*;
        rows.into_par_iter()
            .map(|row| {
                let mut f = HashMap::with_capacity(row.columns.len() + 1);
                f.insert(
                    TABLE_FIELD.to_string(),
                    AmorphicValue::String(table.to_string()),
                );
                for (c, v) in row.columns.into_iter().zip(row.values) {
                    f.insert(c, ast_into_amorphic(v));
                }
                f
            })
            .collect()
    }

    /// Store ingest of prebuilt field maps (one WAL group commit).
    pub fn insert_field_maps(
        &self,
        table: &str,
        batch: Vec<HashMap<String, AmorphicValue>>,
        prep: std::time::Duration,
    ) -> QueryResult<Vec<u64>> {
        let t_lock = std::time::Instant::now();
        let mut store = self.store.write().map_err(lock_error)?;
        let lock_wait = t_lock.elapsed();
        let _columns = self.get_schema_columns(&store, table)?;
        let t0 = std::time::Instant::now();
        let gen_before = store.generation();
        // The table's column snapshot, if current, is taken out of the cache
        // and appended from the batch rows in parallel with the store's
        // index updates and WAL encode (it was a serial step after them).
        // Queries cannot read it meanwhile: they need the store lock we hold.
        let snap = self
            .col_cache
            .lock()
            .ok()
            .and_then(|mut c| match c.get(table) {
                Some((g, _)) if *g == gen_before => c.remove(table).map(|(_, t)| t),
                _ => None,
            });
        let had_snap = snap.is_some();
        let (ids, snap) = store
            .ingest_fields_batch_with(batch, move |rows| {
                snap.and_then(|mut t| std::sync::Arc::make_mut(&mut t).append(rows).then_some(t))
            })
            .map_err(amorphic_error)?;
        let t_cache = std::time::Instant::now();
        match snap {
            Some(t) => {
                if let Ok(mut c) = self.col_cache.lock() {
                    c.insert(table.to_string(), (store.generation(), t));
                }
            }
            // appended snapshot rejected the batch (type change): it stays
            // dropped and is rebuilt on the next query, as before
            None if had_snap => {}
            None => self.refresh_columns_after_load(&store, table, gen_before, &ids),
        }
        if std::env::var_os("JOULEDB_INGEST_TRACE").is_some() {
            eprintln!(
                "insert_rows n={} lock={:?} prep={:?} store={:?} colcache={:?}",
                ids.len(),
                lock_wait,
                prep,
                t_cache - t0,
                t_cache.elapsed()
            );
        }
        if store.pending_holograms() > 0 {
            self.store.mark_pending();
        }
        Ok(ids)
    }

    /// Columnar-first bulk ingest: `batch` (validated typed columns, the
    /// rows the row path would store) is logged to the WAL and committed
    /// (group fsync after the lock is released), appended to the table's
    /// column snapshot, and queued. Row records and indexes are built in
    /// the background, or before any reader or writer that needs them.
    pub fn insert_column_batch(
        &self,
        table: &str,
        mut batch: joule_db_amorphic::column_batch::ColumnBatch,
    ) -> QueryResult<usize> {
        let t_lock = std::time::Instant::now();
        let mut store = self.store.write_bulk().map_err(lock_error)?;
        let lock_wait = t_lock.elapsed();
        let columns = self.get_schema_columns(&store, table)?;
        let t0 = std::time::Instant::now();
        batch.table_field = TABLE_FIELD.to_string();
        batch.table = table.to_string();
        let n = batch.len;

        // Check whether this table is already backed by out-of-core `.jcol` segments,
        // or should transition to `.jcol` segments now to bound RAM on multi-million-row loads.
        let already_disk = self
            .disk_cols
            .read()
            .ok()
            .is_some_and(|dc| dc.contains_key(table));

        let should_spill = if already_disk {
            true
        } else {
            let spill_thresh = crate::disk_col_segments::disk_segment_spill_rows();
            let cached_rows = self
                .col_cache
                .lock()
                .ok()
                .and_then(|c| c.get(table).map(|(_, t)| t.len))
                .unwrap_or(0);
            let existing_rows = cached_rows.max(
                store
                    .records_equal(TABLE_FIELD, &AmorphicValue::String(table.to_string()))
                    .len(),
            );
            if existing_rows.saturating_add(n) > spill_thresh {
                let defs = self
                    .get_column_defs_from_store(&store, table)
                    .unwrap_or_default();
                let has_constraints = defs.iter().any(|d| {
                    d.primary_key
                        || d.unique
                        || d.foreign_key.is_some()
                        || d.computed_expr.is_some()
                });
                let has_btree_idx = store
                    .query_equals(INDEX_TABLE_REF, &AmorphicValue::String(table.to_string()))
                    .records()
                    .iter()
                    .any(|r| {
                        r.get(TABLE_FIELD) == Some(&AmorphicValue::String(INDEX_TABLE.to_string()))
                    });
                !has_constraints && !has_btree_idx
            } else {
                false
            }
        };

        if should_spill {
            batch
                .validate()
                .map_err(|e| QueryError::ExecutionError(format!("invalid column batch: {e}")))?;
            let (drained_batches, drained_recs) = if already_disk {
                (Vec::new(), Vec::new())
            } else {
                store
                    .drain_table_for_disk_spill(TABLE_FIELD, table)
                    .map_err(amorphic_error)?
            };
            let first_id = store.reserve_disk_segment_ids(n);
            drop(store);

            if let Ok(mut c) = self.col_cache.lock() {
                c.remove(table);
            }

            let sync_on = sync_commit_enabled();
            let mut dc_guard = self.disk_cols.write().map_err(lock_error)?;
            let mut dt_owned = match dc_guard.get(table) {
                Some(existing) => (**existing).clone(),
                None => crate::disk_col_segments::DiskColumnTable::create_new(
                    &self.db_path,
                    table,
                    &columns,
                )
                .map_err(|e| QueryError::ExecutionError(format!("create .jcol dir failed: {e}")))?,
            };

            if !drained_recs.is_empty() {
                let mut refs: Vec<&AmorphicRecord> = drained_recs.iter().collect();
                refs.sort_unstable_by_key(|r| r.id);
                dt_owned
                    .append_amorphic_records(&refs, sync_on)
                    .map_err(|e| {
                        QueryError::ExecutionError(format!("spill records to .jcol failed: {e}"))
                    })?;
            }
            for (off, pb) in drained_batches {
                dt_owned
                    .append_column_batch_from(&pb, off, pb.first_id + off as u64, sync_on)
                    .map_err(|e| {
                        QueryError::ExecutionError(format!("spill pending to .jcol failed: {e}"))
                    })?;
            }
            dt_owned
                .append_column_batch_from(&batch, 0, first_id, sync_on)
                .map_err(|e| QueryError::ExecutionError(format!("append .jcol failed: {e}")))?;

            dc_guard.insert(table.to_string(), std::sync::Arc::new(dt_owned));
            drop(dc_guard);

            if std::env::var_os("JOULEDB_INGEST_TRACE").is_some() {
                eprintln!(
                    "insert_columns_disk n={} lock={:?} write_jcol={:?}",
                    n,
                    lock_wait,
                    t0.elapsed()
                );
            }
            return Ok(n);
        }

        let gen_before = store.generation();
        let snap = self
            .col_cache
            .lock()
            .ok()
            .and_then(|mut c| match c.get(table) {
                Some((g, _)) if *g == gen_before => c.remove(table).map(|(_, t)| t),
                _ => None,
            });
        // no snapshot: an empty table's snapshot is just this batch
        let fresh = snap.is_none()
            && store.pending_rows() == 0
            && store
                .records_equal(TABLE_FIELD, &AmorphicValue::String(table.to_string()))
                .is_empty();
        let (_, snap) = store
            .ingest_column_batch_with(batch, move |b| match snap {
                Some(mut t) => std::sync::Arc::make_mut(&mut t)
                    .append_batch(b)
                    .then_some(t),
                None if fresh => Some(std::sync::Arc::new(
                    crate::columnar_exec::TableColumns::from_batch(&columns, b),
                )),
                None => None,
            })
            .map_err(amorphic_error)?;
        if let Some(t) = snap {
            if let Ok(mut c) = self.col_cache.lock() {
                if c.len() > 64 {
                    c.clear();
                }
                c.insert(table.to_string(), (store.generation(), t));
            }
        }
        self.store.mark_columns_pending();
        if std::env::var_os("JOULEDB_INGEST_TRACE").is_some() {
            eprintln!(
                "insert_columns n={} lock={:?} store+colcache={:?}",
                n,
                lock_wait,
                t0.elapsed()
            );
        }
        Ok(n)
    }

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
