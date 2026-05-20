//! MvccTableStorage — MVCC-backed SQL table storage with session management.
//!
//! Bridges the key-value MVCC layer (`MvccTransactionManager`) to SQL operations.
//! Each SQL row is stored as a serialized record under an MVCC key
//! `<table>\0<record_id_be8>`. Active transactions are tracked per session ID.
//!
//! ## Design
//!
//! - **Auto-commit**: When `session_id` is `None`, each DML runs in an implicit
//!   begin→execute→commit cycle (backward-compatible with pre-MVCC behavior).
//! - **Explicit transactions**: `BEGIN` returns a UUID session ID; subsequent
//!   DML with that session ID buffers writes in the MVCC transaction's write set.
//!   `COMMIT` persists to the durable `AmorphicTableStorage`; `ROLLBACK` discards.
//! - **Snapshot isolation**: Concurrent sessions see consistent snapshots.
//!   Write-write conflicts are detected at `put` time (first-committer-wins).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use joule_db_core::concurrency::{
    MvccTransaction, MvccTransactionManager, decode_record_id, encode_record_key,
    encode_table_prefix,
};
use joule_db_query::ast::{Expression, Value as AstValue};
use joule_db_query::error::{QueryError, QueryResult};
use joule_db_query::executor::{RowData, TableStorage};

use crate::amorphic_adapter::AmorphicTableStorage;

/// A serializable record stored as an MVCC value.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MvccRecord {
    columns: Vec<String>,
    values: Vec<AstValue>,
    /// Amorphic store RecordId — present for records bootstrapped from amorphic,
    /// None for records created within an MVCC transaction (not yet persisted).
    #[serde(default)]
    amorphic_id: Option<u64>,
}

impl MvccRecord {
    fn to_row_data(&self) -> RowData {
        RowData::new(self.columns.clone(), self.values.clone())
    }

    fn from_row_data(row: &RowData) -> Self {
        Self {
            columns: row.columns.clone(),
            values: row.values.clone(),
            amorphic_id: None,
        }
    }
}

/// An active transaction session.
/// Snapshot of transaction state at a savepoint
#[derive(Clone)]
struct SavepointSnapshot {
    name: String,
    /// Clone of write_set at savepoint time
    write_set_snapshot: Vec<(Vec<u8>, Option<Vec<u8>>)>,
}

pub struct ActiveSession {
    pub tx: Option<MvccTransaction>,
    pub created_at: Instant,
    savepoints: Vec<SavepointSnapshot>,
}

/// MVCC-backed table storage with session management.
pub struct MvccTableStorage {
    tx_manager: MvccTransactionManager,
    amorphic: Arc<AmorphicTableStorage>,
    sessions: Mutex<HashMap<String, Mutex<ActiveSession>>>,
    next_record_id: AtomicU64,
}

impl MvccTableStorage {
    /// Create a new MVCC adapter wrapping the given amorphic storage.
    pub fn new(amorphic: Arc<AmorphicTableStorage>) -> Self {
        Self {
            tx_manager: MvccTransactionManager::new(),
            amorphic,
            sessions: Mutex::new(HashMap::new()),
            next_record_id: AtomicU64::new(1),
        }
    }

    /// Seed the MVCC store with all existing records from amorphic storage.
    /// Call once at startup so that MVCC reads see pre-existing data.
    /// Tracks amorphic RecordIds so that commit can map deletes/updates correctly.
    pub fn bootstrap_from_amorphic(&self, tables: &[String]) -> QueryResult<usize> {
        let store = self.tx_manager.store();
        let mut total = 0usize;
        let mut max_id = self.next_record_id.load(Ordering::Relaxed);

        for table in tables {
            let rows_with_ids = self.amorphic.scan_with_ids(table)?;
            let columns = self.amorphic.columns(table)?;

            for (amorphic_record_id, row) in &rows_with_ids {
                let record_id = self.next_record_id.fetch_add(1, Ordering::Relaxed);
                if record_id >= max_id {
                    max_id = record_id + 1;
                }
                let key = encode_record_key(table, record_id);
                let mvcc_record = MvccRecord {
                    columns: columns.clone(),
                    values: row.values.clone(),
                    amorphic_id: Some(*amorphic_record_id),
                };
                let value = serde_json::to_vec(&mvcc_record).map_err(|e| {
                    QueryError::ExecutionError(format!("Serialize error during bootstrap: {}", e))
                })?;
                store.insert_committed(&key, value, 0).map_err(|e| {
                    QueryError::ExecutionError(format!("MVCC bootstrap error: {}", e))
                })?;
                total += 1;
            }
        }

        // Ensure next_record_id is past all bootstrapped IDs
        self.next_record_id.store(max_id, Ordering::Relaxed);

        Ok(total)
    }

    // ==================== Session management ====================

    /// Begin a new transaction, returning a UUID session ID.
    pub fn begin_transaction(&self) -> String {
        let session_id = uuid::Uuid::new_v4().to_string();
        let tx = self.tx_manager.begin();
        let session = ActiveSession {
            tx: Some(tx),
            created_at: Instant::now(),
            savepoints: Vec::new(),
        };
        let mut sessions = crate::lock_util::mutex_lock(&self.sessions);
        sessions.insert(session_id.clone(), Mutex::new(session));
        session_id
    }

    /// Commit a transaction. Persists all writes to the durable amorphic store
    /// atomically via a WAL batch transaction, then commits the MVCC transaction.
    ///
    /// The WAL batch is committed BEFORE the MVCC commit. If the process crashes
    /// after WAL commit but before MVCC commit, recovery will replay the WAL entries
    /// and bootstrap_from_amorphic will seed the MVCC store correctly on restart.
    pub fn commit_transaction(&self, session_id: &str) -> QueryResult<()> {
        // Remove the session (acquires sessions lock first, then drops it)
        let session_mutex = {
            let mut sessions = crate::lock_util::mutex_lock(&self.sessions);
            sessions.remove(session_id).ok_or_else(|| {
                QueryError::ExecutionError(format!(
                    "No active transaction for session '{}'",
                    session_id,
                ))
            })?
        };

        let mut session = crate::lock_util::mutex_lock(&session_mutex);
        let tx = session.tx.take().ok_or_else(|| {
            QueryError::ExecutionError("Transaction already consumed".to_string())
        })?;

        // Snapshot the write set BEFORE commit (commit consumes the transaction)
        let write_set: Vec<(Vec<u8>, Option<Vec<u8>>)> = tx
            .write_set()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // If the write set is empty, just commit the MVCC transaction
        if write_set.is_empty() {
            tx.commit()
                .map_err(|e| QueryError::ExecutionError(format!("MVCC commit failed: {}", e)))?;
            return Ok(());
        }

        // Begin atomic WAL batch BEFORE MVCC commit
        self.amorphic.begin_batch()?;

        // Persist all writes within the batch
        let mut batch_ok = true;
        for (key, value) in &write_set {
            if let Some(null_pos) = key.iter().position(|&b| b == 0) {
                let table = std::str::from_utf8(&key[..null_pos]).unwrap_or("");
                if table.is_empty() {
                    continue;
                }

                match value {
                    Some(data) => {
                        // INSERT or UPDATE
                        if let Ok(record) = serde_json::from_slice::<MvccRecord>(data) {
                            let row = record.to_row_data();
                            if let Some(existing_id) = record.amorphic_id {
                                // UPDATE: update fields on existing amorphic record
                                let updates: HashMap<String, joule_db_amorphic::Value> = row
                                    .columns
                                    .iter()
                                    .zip(row.values.iter())
                                    .map(|(col, val)| {
                                        (
                                            col.clone(),
                                            crate::amorphic_adapter::ast_to_amorphic_value(val),
                                        )
                                    })
                                    .collect();
                                if let Err(e) =
                                    self.amorphic.batch_update_record(existing_id, updates)
                                {
                                    eprintln!("WAL batch update error: {}", e);
                                    batch_ok = false;
                                    break;
                                }
                            } else {
                                // INSERT: new record
                                if let Err(e) = self.amorphic.batch_insert(table, &row) {
                                    eprintln!("WAL batch insert error: {}", e);
                                    batch_ok = false;
                                    break;
                                }
                            }
                        }
                    }
                    None => {
                        // DELETE (tombstone) — find the amorphic record ID from the
                        // original MVCC value (before the delete was applied)
                        let record_id_bytes = &key[null_pos + 1..];
                        if record_id_bytes.len() == 8 {
                            let mvcc_record_id =
                                u64::from_be_bytes(record_id_bytes.try_into().unwrap_or([0; 8]));
                            // Look up the original record to get the amorphic_id
                            if let Some(original_data) = self.get_committed_value(key) {
                                if let Ok(original) =
                                    serde_json::from_slice::<MvccRecord>(&original_data)
                                {
                                    if let Some(amorphic_id) = original.amorphic_id {
                                        if let Err(e) =
                                            self.amorphic.batch_delete_record(amorphic_id)
                                        {
                                            eprintln!("WAL batch delete error: {}", e);
                                            batch_ok = false;
                                            break;
                                        }
                                    }
                                }
                            }
                            let _ = mvcc_record_id; // suppress unused warning
                        }
                    }
                }
            }
        }

        if !batch_ok {
            // Rollback the WAL batch
            let _ = self.amorphic.rollback_batch();
            // Rollback the MVCC transaction
            let _ = tx.rollback();
            return Err(QueryError::ExecutionError(
                "Transaction commit failed: WAL batch error".to_string(),
            ));
        }

        // Commit the WAL batch (atomic durability point — fsync)
        if let Err(e) = self.amorphic.commit_batch() {
            let _ = self.amorphic.rollback_batch();
            let _ = tx.rollback();
            return Err(QueryError::ExecutionError(format!(
                "Transaction commit failed: WAL commit error: {}",
                e
            )));
        }

        // Now commit the MVCC transaction (in-memory finalize)
        tx.commit().map_err(|e| {
            // WAL is already committed, so data is durable.
            // MVCC commit failure here is unexpected but data is safe.
            QueryError::ExecutionError(format!("MVCC commit failed (data is durable): {}", e))
        })?;

        Ok(())
    }

    /// Rollback a transaction, discarding all buffered writes.
    pub fn rollback_transaction(&self, session_id: &str) -> QueryResult<()> {
        let session_mutex = {
            let mut sessions = crate::lock_util::mutex_lock(&self.sessions);
            sessions.remove(session_id).ok_or_else(|| {
                QueryError::ExecutionError(format!(
                    "No active transaction for session '{}'",
                    session_id,
                ))
            })?
        };

        let mut session = crate::lock_util::mutex_lock(&session_mutex);
        let tx = session.tx.take().ok_or_else(|| {
            QueryError::ExecutionError("Transaction already consumed".to_string())
        })?;

        tx.rollback()
            .map_err(|e| QueryError::ExecutionError(format!("MVCC rollback failed: {}", e)))?;

        Ok(())
    }

    // ==================== Savepoint operations ====================

    /// Create a savepoint — snapshot the current write_set.
    pub fn create_savepoint(&self, session_id: &str, name: &str) -> Result<(), String> {
        let sessions = crate::lock_util::mutex_lock(&self.sessions);
        if let Some(session_mutex) = sessions.get(session_id) {
            let mut session = crate::lock_util::mutex_lock(session_mutex);
            if let Some(ref tx) = session.tx {
                let snapshot = SavepointSnapshot {
                    name: name.to_string(),
                    write_set_snapshot: tx.write_set_snapshot(),
                };
                session.savepoints.push(snapshot);
                Ok(())
            } else {
                Err("No active transaction".to_string())
            }
        } else {
            // No explicit transaction — savepoint is a no-op (auto-commit mode)
            Ok(())
        }
    }

    /// Rollback to a savepoint — restore the write_set from snapshot.
    pub fn rollback_to_savepoint(&self, session_id: &str, name: &str) -> Result<(), String> {
        let sessions = crate::lock_util::mutex_lock(&self.sessions);
        if let Some(session_mutex) = sessions.get(session_id) {
            let mut session = crate::lock_util::mutex_lock(session_mutex);
            // Find the savepoint (most recent with matching name)
            let idx = session
                .savepoints
                .iter()
                .rposition(|sp| sp.name == name)
                .ok_or_else(|| format!("SAVEPOINT '{}' not found", name))?;
            // Clone the snapshot before truncating
            let snapshot = session.savepoints[idx].clone();
            // Pop all savepoints after this one (they're invalidated)
            session.savepoints.truncate(idx + 1);
            // Restore the transaction's write_set
            if let Some(ref mut tx) = session.tx {
                tx.restore_write_set(snapshot.write_set_snapshot);
            }
            Ok(())
        } else {
            Ok(()) // No explicit transaction
        }
    }

    /// Release a savepoint — discard the snapshot.
    pub fn release_savepoint(&self, session_id: &str, name: &str) -> Result<(), String> {
        let sessions = crate::lock_util::mutex_lock(&self.sessions);
        if let Some(session_mutex) = sessions.get(session_id) {
            let mut session = crate::lock_util::mutex_lock(session_mutex);
            let idx = session
                .savepoints
                .iter()
                .rposition(|sp| sp.name == name)
                .ok_or_else(|| format!("SAVEPOINT '{}' not found", name))?;
            session.savepoints.remove(idx);
            Ok(())
        } else {
            Ok(())
        }
    }

    // ==================== DML operations ====================

    /// Scan all rows for a table, respecting transaction isolation.
    pub fn scan(&self, session_id: Option<&str>, table: &str) -> QueryResult<Vec<RowData>> {
        match session_id {
            Some(sid) => self.with_session(sid, |tx| {
                let prefix = encode_table_prefix(table);
                let entries = tx
                    .scan_prefix(&prefix)
                    .map_err(|e| QueryError::ExecutionError(format!("MVCC scan error: {}", e)))?;
                let mut rows = Vec::with_capacity(entries.len());
                for (_key, data) in entries {
                    let record: MvccRecord = serde_json::from_slice(&data).map_err(|e| {
                        QueryError::ExecutionError(format!("Deserialize error: {}", e))
                    })?;
                    rows.push(record.to_row_data());
                }
                Ok(rows)
            }),
            None => self.amorphic.scan(table),
        }
    }

    /// Time-travel scan: read rows visible at a specific MVCC timestamp.
    pub fn scan_at_timestamp(&self, table: &str, ts: u64) -> QueryResult<Vec<RowData>> {
        let prefix = encode_table_prefix(table);
        let entries = self.tx_manager.scan_prefix_at(&prefix, ts).map_err(|e| {
            QueryError::ExecutionError(format!("MVCC time-travel scan error: {}", e))
        })?;
        let mut rows = Vec::with_capacity(entries.len());
        for (_key, data) in entries {
            let record: MvccRecord = serde_json::from_slice(&data)
                .map_err(|e| QueryError::ExecutionError(format!("Deserialize error: {}", e)))?;
            rows.push(record.to_row_data());
        }
        Ok(rows)
    }

    /// Insert a row into a table.
    pub fn insert(&self, session_id: Option<&str>, table: &str, row: &RowData) -> QueryResult<()> {
        match session_id {
            Some(sid) => self.with_session(sid, |tx| {
                let record_id = self.next_record_id.fetch_add(1, Ordering::Relaxed);
                let key = encode_record_key(table, record_id);
                let record = MvccRecord::from_row_data(row);
                let data = serde_json::to_vec(&record)
                    .map_err(|e| QueryError::ExecutionError(format!("Serialize error: {}", e)))?;
                tx.put(&key, &data)
                    .map_err(|e| QueryError::ExecutionError(format!("MVCC put error: {}", e)))?;
                Ok(())
            }),
            None => self.amorphic.insert(table, row),
        }
    }

    /// Update matching rows. Returns the number of updated rows.
    pub fn update(
        &self,
        session_id: Option<&str>,
        table: &str,
        assignments: &HashMap<String, AstValue>,
        predicate: Option<&Expression>,
    ) -> QueryResult<usize> {
        match session_id {
            Some(sid) => {
                // Scan, filter, update in MVCC
                let prefix = encode_table_prefix(table);
                let entries = self.with_session(sid, |tx| {
                    tx.scan_prefix(&prefix)
                        .map_err(|e| QueryError::ExecutionError(format!("MVCC scan error: {}", e)))
                })?;

                let mut updated = 0;
                for (key, data) in entries {
                    let record: MvccRecord = serde_json::from_slice(&data).map_err(|e| {
                        QueryError::ExecutionError(format!("Deserialize error: {}", e))
                    })?;
                    let row = record.to_row_data();
                    let matches = predicate
                        .map(|p| evaluate_predicate(&row, p))
                        .unwrap_or(true);

                    if matches {
                        let mut new_values = row.values.clone();
                        for (col, val) in assignments {
                            if let Some(idx) = row.columns.iter().position(|c| c == col) {
                                new_values[idx] = val.clone();
                            }
                        }
                        let new_record = MvccRecord {
                            columns: row.columns.clone(),
                            values: new_values,
                            amorphic_id: record.amorphic_id,
                        };
                        let new_data = serde_json::to_vec(&new_record).map_err(|e| {
                            QueryError::ExecutionError(format!("Serialize error: {}", e))
                        })?;
                        self.with_session(sid, |tx| {
                            tx.put(&key, &new_data).map_err(|e| {
                                QueryError::ExecutionError(format!("MVCC put error: {}", e))
                            })
                        })?;
                        updated += 1;
                    }
                }
                Ok(updated)
            }
            None => self.amorphic.update(table, assignments, predicate),
        }
    }

    /// Update matching rows by evaluating SET expressions per-row.
    /// This allows expressions like `SET age = age + 1` to reference column values.
    pub fn update_with_expressions(
        &self,
        session_id: Option<&str>,
        table: &str,
        assignments: &[(String, Expression)],
        predicate: Option<&Expression>,
        params: &[serde_json::Value],
    ) -> QueryResult<usize> {
        match session_id {
            Some(sid) => {
                let prefix = encode_table_prefix(table);
                let entries = self.with_session(sid, |tx| {
                    tx.scan_prefix(&prefix)
                        .map_err(|e| QueryError::ExecutionError(format!("MVCC scan error: {}", e)))
                })?;

                let mut updated = 0;
                for (key, data) in entries {
                    let record: MvccRecord = serde_json::from_slice(&data).map_err(|e| {
                        QueryError::ExecutionError(format!("Deserialize error: {}", e))
                    })?;
                    let row = record.to_row_data();
                    let matches = predicate
                        .map(|p| evaluate_predicate(&row, p))
                        .unwrap_or(true);

                    if matches {
                        // Convert row to json for expression evaluation
                        let json_row: Vec<serde_json::Value> = row
                            .values
                            .iter()
                            .map(|v| crate::json_ops::ast_value_to_json(v))
                            .collect();

                        // Evaluate each SET expression with row context
                        let mut new_values = row.values.clone();
                        for (col, expr) in assignments {
                            if let Some(idx) = row.columns.iter().position(|c| c == col) {
                                let json_val = crate::query::evaluate_expression(
                                    expr,
                                    &row.columns,
                                    &json_row,
                                    params,
                                );
                                new_values[idx] = crate::json_ops::json_to_ast_value(&json_val);
                            }
                        }

                        let new_record = MvccRecord {
                            columns: row.columns.clone(),
                            values: new_values,
                            amorphic_id: record.amorphic_id,
                        };
                        let new_data = serde_json::to_vec(&new_record).map_err(|e| {
                            QueryError::ExecutionError(format!("Serialize error: {}", e))
                        })?;
                        self.with_session(sid, |tx| {
                            tx.put(&key, &new_data).map_err(|e| {
                                QueryError::ExecutionError(format!("MVCC put error: {}", e))
                            })
                        })?;
                        updated += 1;
                    }
                }
                Ok(updated)
            }
            None => self
                .amorphic
                .update_with_expressions(table, assignments, predicate, params),
        }
    }

    /// Delete matching rows. Returns the number of deleted rows.
    pub fn delete(
        &self,
        session_id: Option<&str>,
        table: &str,
        predicate: Option<&Expression>,
    ) -> QueryResult<usize> {
        match session_id {
            Some(sid) => {
                let prefix = encode_table_prefix(table);
                let entries = self.with_session(sid, |tx| {
                    tx.scan_prefix(&prefix)
                        .map_err(|e| QueryError::ExecutionError(format!("MVCC scan error: {}", e)))
                })?;

                let mut deleted = 0;
                for (key, data) in entries {
                    let record: MvccRecord = serde_json::from_slice(&data).map_err(|e| {
                        QueryError::ExecutionError(format!("Deserialize error: {}", e))
                    })?;
                    let row = record.to_row_data();
                    let matches = predicate
                        .map(|p| evaluate_predicate(&row, p))
                        .unwrap_or(true);

                    if matches {
                        self.with_session(sid, |tx| {
                            tx.delete(&key).map_err(|e| {
                                QueryError::ExecutionError(format!("MVCC delete error: {}", e))
                            })?;
                            Ok(())
                        })?;
                        deleted += 1;
                    }
                }
                Ok(deleted)
            }
            None => self.amorphic.delete(table, predicate),
        }
    }

    /// Like update_with_expressions, but also returns the updated rows as JSON.
    pub fn update_with_expressions_returning(
        &self,
        session_id: Option<&str>,
        table: &str,
        assignments: &[(String, Expression)],
        predicate: Option<&Expression>,
        params: &[serde_json::Value],
    ) -> QueryResult<(usize, Vec<Vec<serde_json::Value>>)> {
        match session_id {
            Some(sid) => {
                let prefix = encode_table_prefix(table);
                let entries = self.with_session(sid, |tx| {
                    tx.scan_prefix(&prefix)
                        .map_err(|e| QueryError::ExecutionError(format!("MVCC scan error: {}", e)))
                })?;

                let mut updated = 0;
                let mut returned_rows = Vec::new();
                for (key, data) in entries {
                    let record: MvccRecord = serde_json::from_slice(&data).map_err(|e| {
                        QueryError::ExecutionError(format!("Deserialize error: {}", e))
                    })?;
                    let row = record.to_row_data();
                    let matches = predicate
                        .map(|p| evaluate_predicate(&row, p))
                        .unwrap_or(true);

                    if matches {
                        let json_row: Vec<serde_json::Value> = row
                            .values
                            .iter()
                            .map(|v| crate::json_ops::ast_value_to_json(v))
                            .collect();

                        let mut new_values = row.values.clone();
                        for (col, expr) in assignments {
                            if let Some(idx) = row.columns.iter().position(|c| c == col) {
                                let json_val = crate::query::evaluate_expression(
                                    expr,
                                    &row.columns,
                                    &json_row,
                                    params,
                                );
                                new_values[idx] = crate::json_ops::json_to_ast_value(&json_val);
                            }
                        }

                        let new_record = MvccRecord {
                            columns: row.columns.clone(),
                            values: new_values.clone(),
                            amorphic_id: record.amorphic_id,
                        };
                        let new_data = serde_json::to_vec(&new_record).map_err(|e| {
                            QueryError::ExecutionError(format!("Serialize error: {}", e))
                        })?;
                        self.with_session(sid, |tx| {
                            tx.put(&key, &new_data).map_err(|e| {
                                QueryError::ExecutionError(format!("MVCC put error: {}", e))
                            })
                        })?;

                        let updated_json: Vec<serde_json::Value> = new_values
                            .iter()
                            .map(|v| crate::json_ops::ast_value_to_json(v))
                            .collect();
                        returned_rows.push(updated_json);
                        updated += 1;
                    }
                }
                Ok((updated, returned_rows))
            }
            None => self.amorphic.update_with_expressions_returning(
                table,
                assignments,
                predicate,
                params,
            ),
        }
    }

    /// Delete matching rows and return the deleted rows as JSON.
    pub fn delete_returning(
        &self,
        session_id: Option<&str>,
        table: &str,
        predicate: Option<&Expression>,
    ) -> QueryResult<(usize, Vec<Vec<serde_json::Value>>)> {
        match session_id {
            Some(sid) => {
                let prefix = encode_table_prefix(table);
                let entries = self.with_session(sid, |tx| {
                    tx.scan_prefix(&prefix)
                        .map_err(|e| QueryError::ExecutionError(format!("MVCC scan error: {}", e)))
                })?;

                let mut deleted = 0;
                let mut returned_rows = Vec::new();
                for (key, data) in entries {
                    let record: MvccRecord = serde_json::from_slice(&data).map_err(|e| {
                        QueryError::ExecutionError(format!("Deserialize error: {}", e))
                    })?;
                    let row = record.to_row_data();
                    let matches = predicate
                        .map(|p| evaluate_predicate(&row, p))
                        .unwrap_or(true);

                    if matches {
                        let json_row: Vec<serde_json::Value> = row
                            .values
                            .iter()
                            .map(|v| crate::json_ops::ast_value_to_json(v))
                            .collect();

                        self.with_session(sid, |tx| {
                            tx.delete(&key).map_err(|e| {
                                QueryError::ExecutionError(format!("MVCC delete error: {}", e))
                            })?;
                            Ok(())
                        })?;
                        returned_rows.push(json_row);
                        deleted += 1;
                    }
                }
                Ok((deleted, returned_rows))
            }
            None => self.amorphic.delete_returning(table, predicate),
        }
    }

    // ==================== Session reaper ====================

    /// Clean up sessions that have been idle longer than `timeout`.
    /// Returns the number of sessions cleaned up.
    pub fn cleanup_stale_sessions(&self, timeout: std::time::Duration) -> usize {
        let mut sessions = crate::lock_util::mutex_lock(&self.sessions);
        let stale: Vec<String> = sessions
            .iter()
            .filter_map(|(id, mutex)| {
                let session = mutex.lock().ok()?;
                if session.created_at.elapsed() > timeout {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();

        let count = stale.len();
        for id in stale {
            if let Some(session_mutex) = sessions.remove(&id) {
                if let Ok(mut session) = session_mutex.lock() {
                    if let Some(tx) = session.tx.take() {
                        let _ = tx.rollback();
                    }
                }
            }
        }
        count
    }

    /// Number of active sessions.
    pub fn active_session_count(&self) -> usize {
        crate::lock_util::mutex_lock(&self.sessions).len()
    }

    // ==================== Internal helpers ====================

    /// Look up the most recent committed value for a key in the MVCC store.
    /// Used to find the amorphic_id for a record being deleted.
    fn get_committed_value(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.tx_manager.get(key)
    }

    /// Execute a closure with the MVCC transaction for a session.
    fn with_session<F, T>(&self, session_id: &str, f: F) -> QueryResult<T>
    where
        F: FnOnce(&mut MvccTransaction) -> QueryResult<T>,
    {
        let sessions = crate::lock_util::mutex_lock(&self.sessions);
        let session_mutex = sessions.get(session_id).ok_or_else(|| {
            QueryError::ExecutionError(format!(
                "No active transaction for session '{}'",
                session_id,
            ))
        })?;
        let mut session = crate::lock_util::mutex_lock(&session_mutex);
        let tx = session.tx.as_mut().ok_or_else(|| {
            QueryError::ExecutionError("Transaction already consumed".to_string())
        })?;
        f(tx)
    }
}

/// Minimal predicate evaluator for MVCC-side filtering.
/// Handles column = literal comparisons and AND/OR.
fn evaluate_predicate(row: &RowData, expr: &Expression) -> bool {
    match expr {
        Expression::Binary { left, op, right } => {
            use joule_db_query::ast::Operator;
            match op {
                Operator::And => evaluate_predicate(row, left) && evaluate_predicate(row, right),
                Operator::Or => evaluate_predicate(row, left) || evaluate_predicate(row, right),
                _ => {
                    let left_val = eval_expr(row, left);
                    let right_val = eval_expr(row, right);
                    match op {
                        Operator::Eq => left_val == right_val,
                        Operator::Ne => left_val != right_val,
                        _ => true, // Other ops pass through
                    }
                }
            }
        }
        _ => true,
    }
}

fn eval_expr(row: &RowData, expr: &Expression) -> AstValue {
    match expr {
        Expression::Column(name) => row.get(name).cloned().unwrap_or(AstValue::Null),
        Expression::Literal(v) => v.clone(),
        _ => AstValue::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use joule_db_amorphic::DurableAmorphicStore;
    use joule_db_query::executor::TableStorage;
    use tempfile::TempDir;

    fn setup() -> (TempDir, Arc<AmorphicTableStorage>, MvccTableStorage) {
        let dir = TempDir::new().unwrap();
        let store = DurableAmorphicStore::open(dir.path()).unwrap();
        let amorphic = Arc::new(AmorphicTableStorage::new(store));
        let mvcc = MvccTableStorage::new(amorphic.clone());
        (dir, amorphic, mvcc)
    }

    fn setup_with_table(
        table: &str,
        columns: &[&str],
    ) -> (TempDir, Arc<AmorphicTableStorage>, MvccTableStorage) {
        let (dir, amorphic, mvcc) = setup();
        let cols: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        amorphic.create_table(table, &cols).unwrap();
        (dir, amorphic, mvcc)
    }

    #[test]
    fn test_auto_commit_insert_and_scan() {
        let (_dir, amorphic, mvcc) = setup_with_table("users", &["name", "age"]);

        // Auto-commit insert (session_id = None)
        let row = RowData::new(
            vec!["name".into(), "age".into()],
            vec![AstValue::String("Alice".into()), AstValue::Int(30)],
        );
        mvcc.insert(None, "users", &row).unwrap();

        // Auto-commit scan
        let rows = mvcc.scan(None, "users").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("name"), Some(&AstValue::String("Alice".into())));
    }

    #[test]
    fn test_begin_commit() {
        let (_dir, amorphic, mvcc) = setup_with_table("users", &["name", "age"]);

        let sid = mvcc.begin_transaction();

        // Insert within transaction
        let row = RowData::new(
            vec!["name".into(), "age".into()],
            vec![AstValue::String("Bob".into()), AstValue::Int(25)],
        );
        mvcc.insert(Some(&sid), "users", &row).unwrap();

        // Visible within same session
        let rows = mvcc.scan(Some(&sid), "users").unwrap();
        assert_eq!(rows.len(), 1);

        // Not visible via auto-commit (amorphic) until committed
        let amorphic_rows = amorphic.scan("users").unwrap();
        assert_eq!(amorphic_rows.len(), 0);

        // Commit
        mvcc.commit_transaction(&sid).unwrap();

        // Now visible via amorphic
        let amorphic_rows = amorphic.scan("users").unwrap();
        assert_eq!(amorphic_rows.len(), 1);
    }

    #[test]
    fn test_begin_rollback() {
        let (_dir, _amorphic, mvcc) = setup_with_table("users", &["name"]);

        let sid = mvcc.begin_transaction();

        let row = RowData::new(vec!["name".into()], vec![AstValue::String("Temp".into())]);
        mvcc.insert(Some(&sid), "users", &row).unwrap();

        // Visible in session
        let rows = mvcc.scan(Some(&sid), "users").unwrap();
        assert_eq!(rows.len(), 1);

        // Rollback
        mvcc.rollback_transaction(&sid).unwrap();

        // Session gone
        assert!(mvcc.scan(Some(&sid), "users").is_err());
    }

    #[test]
    fn test_snapshot_isolation() {
        let (_dir, _amorphic, mvcc) = setup_with_table("t", &["val"]);

        // Insert initial data via auto-commit through amorphic
        let row = RowData::new(vec!["val".into()], vec![AstValue::Int(1)]);
        mvcc.insert(None, "t", &row).unwrap();

        // Session A starts — sees 1 row (via amorphic since no MVCC data yet)
        let sid_a = mvcc.begin_transaction();

        // Bootstrap MVCC so session A can see the data
        mvcc.bootstrap_from_amorphic(&["t".to_string()]).unwrap();

        // Session B starts
        let sid_b = mvcc.begin_transaction();

        // Session A inserts
        let row2 = RowData::new(vec!["val".into()], vec![AstValue::Int(2)]);
        mvcc.insert(Some(&sid_a), "t", &row2).unwrap();
        mvcc.commit_transaction(&sid_a).unwrap();

        // Session B should NOT see session A's insert (snapshot isolation)
        let rows_b = mvcc.scan(Some(&sid_b), "t").unwrap();
        assert_eq!(rows_b.len(), 1); // Only the bootstrapped row
        assert_eq!(rows_b[0].get("val"), Some(&AstValue::Int(1)));

        mvcc.rollback_transaction(&sid_b).unwrap();
    }

    #[test]
    fn test_write_write_conflict() {
        let (_dir, _amorphic, mvcc) = setup_with_table("t", &["val"]);

        // Bootstrap a record
        mvcc.bootstrap_from_amorphic(&["t".to_string()]).ok();

        let sid_a = mvcc.begin_transaction();
        let sid_b = mvcc.begin_transaction();

        // Session A inserts
        let row = RowData::new(vec!["val".into()], vec![AstValue::Int(1)]);
        mvcc.insert(Some(&sid_a), "t", &row).unwrap();

        // Session B inserts different key — no conflict (different record IDs)
        let row2 = RowData::new(vec!["val".into()], vec![AstValue::Int(2)]);
        mvcc.insert(Some(&sid_b), "t", &row2).unwrap();

        // Both can commit (different keys)
        mvcc.commit_transaction(&sid_a).unwrap();
        mvcc.commit_transaction(&sid_b).unwrap();
    }

    #[test]
    fn test_bootstrap_from_amorphic() {
        let (_dir, amorphic, mvcc) = setup_with_table("users", &["name", "age"]);

        // Insert data via amorphic
        amorphic
            .insert(
                "users",
                &RowData::new(
                    vec!["name".into(), "age".into()],
                    vec![AstValue::String("Alice".into()), AstValue::Int(30)],
                ),
            )
            .unwrap();
        amorphic
            .insert(
                "users",
                &RowData::new(
                    vec!["name".into(), "age".into()],
                    vec![AstValue::String("Bob".into()), AstValue::Int(25)],
                ),
            )
            .unwrap();

        // Bootstrap
        let count = mvcc
            .bootstrap_from_amorphic(&["users".to_string()])
            .unwrap();
        assert_eq!(count, 2);

        // MVCC transaction should see bootstrapped data
        let sid = mvcc.begin_transaction();
        let rows = mvcc.scan(Some(&sid), "users").unwrap();
        assert_eq!(rows.len(), 2);
        mvcc.rollback_transaction(&sid).unwrap();
    }

    #[test]
    fn test_cleanup_stale_sessions() {
        let (_dir, _amorphic, mvcc) = setup_with_table("t", &["x"]);

        let _sid1 = mvcc.begin_transaction();
        let _sid2 = mvcc.begin_transaction();

        assert_eq!(mvcc.active_session_count(), 2);

        // With zero timeout, all sessions are stale
        let cleaned = mvcc.cleanup_stale_sessions(std::time::Duration::ZERO);
        assert_eq!(cleaned, 2);
        assert_eq!(mvcc.active_session_count(), 0);
    }

    #[test]
    fn test_commit_without_session_errors() {
        let (_dir, _amorphic, mvcc) = setup();

        let result = mvcc.commit_transaction("nonexistent");
        assert!(result.is_err());
    }

    // ==================== Durability tests ====================

    /// Helper: create a fresh MvccTableStorage from an existing directory path
    /// (simulates reopening after crash/restart).
    fn reopen_from_path(path: &std::path::Path) -> (Arc<AmorphicTableStorage>, MvccTableStorage) {
        let store = DurableAmorphicStore::open(path).unwrap();
        let amorphic = Arc::new(AmorphicTableStorage::new(store));
        let mvcc = MvccTableStorage::new(amorphic.clone());
        (amorphic, mvcc)
    }

    #[test]
    fn test_explicit_tx_commit_durable() {
        let dir = TempDir::new().unwrap();

        // Phase 1: BEGIN → INSERT → COMMIT
        {
            let store = DurableAmorphicStore::open(dir.path()).unwrap();
            let amorphic = Arc::new(AmorphicTableStorage::new(store));
            let mvcc = MvccTableStorage::new(amorphic.clone());

            amorphic
                .create_table("users", &["name".into(), "age".into()])
                .unwrap();

            let sid = mvcc.begin_transaction();
            let row = RowData::new(
                vec!["name".into(), "age".into()],
                vec![AstValue::String("Alice".into()), AstValue::Int(30)],
            );
            mvcc.insert(Some(&sid), "users", &row).unwrap();
            mvcc.commit_transaction(&sid).unwrap();
        }

        // Phase 2: Reopen from same path → verify data present
        {
            let (amorphic, _mvcc) = reopen_from_path(dir.path());
            let rows = amorphic.scan("users").unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].get("name"), Some(&AstValue::String("Alice".into())));
            assert_eq!(rows[0].get("age"), Some(&AstValue::Int(30)));
        }
    }

    #[test]
    fn test_explicit_tx_rollback_not_durable() {
        let dir = TempDir::new().unwrap();

        // Phase 1: BEGIN → INSERT → ROLLBACK
        {
            let store = DurableAmorphicStore::open(dir.path()).unwrap();
            let amorphic = Arc::new(AmorphicTableStorage::new(store));
            let mvcc = MvccTableStorage::new(amorphic.clone());

            amorphic.create_table("users", &["name".into()]).unwrap();

            let sid = mvcc.begin_transaction();
            let row = RowData::new(vec!["name".into()], vec![AstValue::String("Ghost".into())]);
            mvcc.insert(Some(&sid), "users", &row).unwrap();
            mvcc.rollback_transaction(&sid).unwrap();
        }

        // Phase 2: Reopen → verify data absent
        {
            let (amorphic, _mvcc) = reopen_from_path(dir.path());
            let rows = amorphic.scan("users").unwrap();
            assert_eq!(rows.len(), 0);
        }
    }

    #[test]
    fn test_crash_before_commit_loses_data() {
        let dir = TempDir::new().unwrap();

        // Phase 1: BEGIN → INSERT (NO COMMIT) → drop (simulates crash)
        {
            let store = DurableAmorphicStore::open(dir.path()).unwrap();
            let amorphic = Arc::new(AmorphicTableStorage::new(store));
            let mvcc = MvccTableStorage::new(amorphic.clone());

            amorphic.create_table("users", &["name".into()]).unwrap();

            let sid = mvcc.begin_transaction();
            let row = RowData::new(vec!["name".into()], vec![AstValue::String("Lost".into())]);
            mvcc.insert(Some(&sid), "users", &row).unwrap();
            // NO COMMIT — drop simulates crash
        }

        // Phase 2: Reopen → verify data absent
        {
            let (amorphic, _mvcc) = reopen_from_path(dir.path());
            let rows = amorphic.scan("users").unwrap();
            assert_eq!(rows.len(), 0);
        }
    }

    #[test]
    fn test_multi_row_transaction_atomicity() {
        let dir = TempDir::new().unwrap();

        // Phase 1: BEGIN → INSERT 3 rows → COMMIT
        {
            let store = DurableAmorphicStore::open(dir.path()).unwrap();
            let amorphic = Arc::new(AmorphicTableStorage::new(store));
            let mvcc = MvccTableStorage::new(amorphic.clone());

            amorphic
                .create_table("items", &["name".into(), "qty".into()])
                .unwrap();

            let sid = mvcc.begin_transaction();
            for (name, qty) in &[("Apple", 10), ("Banana", 20), ("Cherry", 30)] {
                let row = RowData::new(
                    vec!["name".into(), "qty".into()],
                    vec![AstValue::String(name.to_string()), AstValue::Int(*qty)],
                );
                mvcc.insert(Some(&sid), "items", &row).unwrap();
            }
            mvcc.commit_transaction(&sid).unwrap();
        }

        // Phase 2: Reopen → all 3 rows present
        {
            let (amorphic, _mvcc) = reopen_from_path(dir.path());
            let rows = amorphic.scan("items").unwrap();
            assert_eq!(rows.len(), 3);
        }
    }

    #[test]
    fn test_delete_in_transaction_durable() {
        let dir = TempDir::new().unwrap();

        // Phase 1: Auto-commit INSERT → BEGIN → DELETE → COMMIT
        {
            let store = DurableAmorphicStore::open(dir.path()).unwrap();
            let amorphic = Arc::new(AmorphicTableStorage::new(store));
            let mvcc = MvccTableStorage::new(amorphic.clone());

            amorphic.create_table("t", &["name".into()]).unwrap();

            // Auto-commit insert (goes directly to amorphic)
            let row = RowData::new(
                vec!["name".into()],
                vec![AstValue::String("ToDelete".into())],
            );
            mvcc.insert(None, "t", &row).unwrap();

            // Bootstrap MVCC from amorphic so transaction can see the row
            mvcc.bootstrap_from_amorphic(&["t".to_string()]).unwrap();

            // Now delete within a transaction
            let sid = mvcc.begin_transaction();
            let pred = Expression::Binary {
                left: Box::new(Expression::Column("name".into())),
                op: joule_db_query::ast::Operator::Eq,
                right: Box::new(Expression::Literal(AstValue::String("ToDelete".into()))),
            };
            let deleted = mvcc.delete(Some(&sid), "t", Some(&pred)).unwrap();
            assert_eq!(deleted, 1);
            mvcc.commit_transaction(&sid).unwrap();
        }

        // Phase 2: Reopen → verify record deleted
        {
            let (amorphic, _mvcc) = reopen_from_path(dir.path());
            let rows = amorphic.scan("t").unwrap();
            assert_eq!(rows.len(), 0, "Deleted record should not survive restart");
        }
    }

    #[test]
    fn test_update_in_transaction_durable() {
        let dir = TempDir::new().unwrap();

        // Phase 1: Auto-commit INSERT → BEGIN → UPDATE → COMMIT
        {
            let store = DurableAmorphicStore::open(dir.path()).unwrap();
            let amorphic = Arc::new(AmorphicTableStorage::new(store));
            let mvcc = MvccTableStorage::new(amorphic.clone());

            amorphic
                .create_table("t", &["name".into(), "val".into()])
                .unwrap();

            let row = RowData::new(
                vec!["name".into(), "val".into()],
                vec![AstValue::String("key".into()), AstValue::Int(1)],
            );
            mvcc.insert(None, "t", &row).unwrap();

            // Bootstrap so transaction can see the row
            mvcc.bootstrap_from_amorphic(&["t".to_string()]).unwrap();

            // Update within transaction
            let sid = mvcc.begin_transaction();
            let mut assignments = HashMap::new();
            assignments.insert("val".to_string(), AstValue::Int(42));
            let pred = Expression::Binary {
                left: Box::new(Expression::Column("name".into())),
                op: joule_db_query::ast::Operator::Eq,
                right: Box::new(Expression::Literal(AstValue::String("key".into()))),
            };
            let updated = mvcc
                .update(Some(&sid), "t", &assignments, Some(&pred))
                .unwrap();
            assert_eq!(updated, 1);
            mvcc.commit_transaction(&sid).unwrap();
        }

        // Phase 2: Reopen → verify updated values
        {
            let (amorphic, _mvcc) = reopen_from_path(dir.path());
            let rows = amorphic.scan("t").unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].get("val"), Some(&AstValue::Int(42)));
        }
    }

    #[test]
    fn test_multi_table_tx_atomicity() {
        let dir = TempDir::new().unwrap();

        // Phase 1: BEGIN → INSERT into two tables → COMMIT
        {
            let store = DurableAmorphicStore::open(dir.path()).unwrap();
            let amorphic = Arc::new(AmorphicTableStorage::new(store));
            let mvcc = MvccTableStorage::new(amorphic.clone());

            amorphic.create_table("orders", &["id".into()]).unwrap();
            amorphic
                .create_table("items", &["order_id".into(), "product".into()])
                .unwrap();

            let sid = mvcc.begin_transaction();

            let order_row = RowData::new(vec!["id".into()], vec![AstValue::Int(1)]);
            mvcc.insert(Some(&sid), "orders", &order_row).unwrap();

            let item_row = RowData::new(
                vec!["order_id".into(), "product".into()],
                vec![AstValue::Int(1), AstValue::String("Widget".into())],
            );
            mvcc.insert(Some(&sid), "items", &item_row).unwrap();

            mvcc.commit_transaction(&sid).unwrap();
        }

        // Phase 2: Reopen → both tables have data
        {
            let (amorphic, _mvcc) = reopen_from_path(dir.path());
            let orders = amorphic.scan("orders").unwrap();
            assert_eq!(orders.len(), 1);
            let items = amorphic.scan("items").unwrap();
            assert_eq!(items.len(), 1);
            assert_eq!(
                items[0].get("product"),
                Some(&AstValue::String("Widget".into()))
            );
        }
    }

    #[test]
    fn test_auto_commit_still_works() {
        let dir = TempDir::new().unwrap();

        // Phase 1: INSERT without BEGIN → drop
        {
            let store = DurableAmorphicStore::open(dir.path()).unwrap();
            let amorphic = Arc::new(AmorphicTableStorage::new(store));
            let mvcc = MvccTableStorage::new(amorphic.clone());

            amorphic.create_table("t", &["val".into()]).unwrap();

            let row = RowData::new(vec!["val".into()], vec![AstValue::Int(99)]);
            mvcc.insert(None, "t", &row).unwrap();
        }

        // Phase 2: Reopen → verify data present (regression test)
        {
            let (amorphic, _mvcc) = reopen_from_path(dir.path());
            let rows = amorphic.scan("t").unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].get("val"), Some(&AstValue::Int(99)));
        }
    }

    #[test]
    fn test_update_in_transaction() {
        let (_dir, _amorphic, mvcc) = setup_with_table("t", &["name", "val"]);

        // Insert via transaction
        let sid = mvcc.begin_transaction();
        let row = RowData::new(
            vec!["name".into(), "val".into()],
            vec![AstValue::String("a".into()), AstValue::Int(1)],
        );
        mvcc.insert(Some(&sid), "t", &row).unwrap();

        // Update within same transaction
        let mut assignments = HashMap::new();
        assignments.insert("val".to_string(), AstValue::Int(42));
        let pred = Expression::Binary {
            left: Box::new(Expression::Column("name".into())),
            op: joule_db_query::ast::Operator::Eq,
            right: Box::new(Expression::Literal(AstValue::String("a".into()))),
        };
        let updated = mvcc
            .update(Some(&sid), "t", &assignments, Some(&pred))
            .unwrap();
        assert_eq!(updated, 1);

        // Verify update visible in session
        let rows = mvcc.scan(Some(&sid), "t").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("val"), Some(&AstValue::Int(42)));

        mvcc.commit_transaction(&sid).unwrap();
    }

    #[test]
    fn test_delete_in_transaction() {
        let (_dir, _amorphic, mvcc) = setup_with_table("t", &["name"]);

        // Insert two rows
        let sid = mvcc.begin_transaction();
        mvcc.insert(
            Some(&sid),
            "t",
            &RowData::new(vec!["name".into()], vec![AstValue::String("keep".into())]),
        )
        .unwrap();
        mvcc.insert(
            Some(&sid),
            "t",
            &RowData::new(vec!["name".into()], vec![AstValue::String("remove".into())]),
        )
        .unwrap();

        // Delete one
        let pred = Expression::Binary {
            left: Box::new(Expression::Column("name".into())),
            op: joule_db_query::ast::Operator::Eq,
            right: Box::new(Expression::Literal(AstValue::String("remove".into()))),
        };
        let deleted = mvcc.delete(Some(&sid), "t", Some(&pred)).unwrap();
        assert_eq!(deleted, 1);

        // Verify
        let rows = mvcc.scan(Some(&sid), "t").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("name"), Some(&AstValue::String("keep".into())));

        mvcc.commit_transaction(&sid).unwrap();
    }
}
