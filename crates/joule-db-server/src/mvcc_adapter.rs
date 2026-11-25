//! MvccTableStorage — MVCC-backed SQL table storage with session management.
//!
//! Bridges the key-value MVCC layer (`MvccTransactionManager`) to SQL operations.
//! Each SQL row is stored as a serialized record under an MVCC key
//! `<table>\0<record_id_be8>`. Active transactions are tracked per session ID.
//!
//! ## Design
//!
//! - **Auto-commit**: When `session_id` is `None`, each DML runs directly on
//!   `AmorphicTableStorage` and keeps MVCC metadata synchronized for active snapshots.
//! - **Lazy & Index-Assisted MVCC Seeding**: Pre-existing `AmorphicTableStorage` rows
//!   (including rows loaded via streaming `COPY ... FROM` or auto-commit `INSERT` after
//!   server startup) are seeded on demand—either per matching row via `O(1)` secondary
//!   index lookup (`WHERE col = literal`) or per table on full-table scans.
//! - **Atomic Group-Committed Durability**: `COMMIT` validates SI/SSI conflicts BEFORE
//!   persisting, executes the WAL batch under a single exclusive store lock (preventing
//!   concurrent `"Batch already active"` races), records durable `amorphic_id`s on newly
//!   inserted MVCC records (preventing duplicate rows on subsequent `UPDATE`), and
//!   coalesces hardware `fsync` across concurrent committers via `GroupSync`.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use joule_db_core::concurrency::{
    MvccTransaction, MvccTransactionManager, advance_mvcc_timestamp, decode_record_id,
    encode_record_key, encode_table_prefix,
};
use joule_db_query::ast::{Expression, Value as AstValue};
use joule_db_query::error::{QueryError, QueryResult};
use joule_db_query::executor::{RowData, TableStorage};

use crate::amorphic_adapter::{
    AmorphicTableStorage, AtomicBatchOp, DeferredBatchSync, evaluate_mvcc_predicate,
};

/// Base offset for temporary MVCC record IDs assigned to uncommitted `INSERT` rows,
/// guaranteeing zero collision with `AmorphicStore` `RecordId`s (`1, 2, ...`).
const NEW_RECORD_ID_BASE: u64 = 1u64 << 62;

/// A serializable record stored as an MVCC value.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MvccRecord {
    columns: Vec<String>,
    values: Vec<AstValue>,
    /// Amorphic store RecordId — present for records persisted in amorphic,
    /// None only while an uncommitted INSERT has not yet been committed.
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

/// Snapshot of transaction state at a savepoint.
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

/// MVCC-backed table storage with session management and lazy Amorphic synchronization.
pub struct MvccTableStorage {
    tx_manager: MvccTransactionManager,
    amorphic: Arc<AmorphicTableStorage>,
    sessions: Mutex<HashMap<String, Mutex<ActiveSession>>>,
    next_temp_id: AtomicU64,
    /// Per-table mapping `amorphic_record_id -> mvcc_record_id` for all records seeded into MVCC.
    table_id_map: Mutex<HashMap<String, HashMap<u64, u64>>>,
    /// Per-table set of `mvcc_record_id`s deleted via MVCC (so older snapshots still find them).
    mvcc_deleted_ids: Mutex<HashMap<String, HashSet<u64>>>,
    /// Per-(table, column) set of `mvcc_record_id`s whose `column` value was modified via MVCC.
    mvcc_col_updated_ids: Mutex<HashMap<(String, String), HashSet<u64>>>,
}

impl MvccTableStorage {
    /// Create a new MVCC adapter wrapping the given amorphic storage.
    pub fn new(amorphic: Arc<AmorphicTableStorage>) -> Self {
        Self {
            tx_manager: MvccTransactionManager::new(),
            amorphic,
            sessions: Mutex::new(HashMap::new()),
            next_temp_id: AtomicU64::new(NEW_RECORD_ID_BASE),
            table_id_map: Mutex::new(HashMap::new()),
            mvcc_deleted_ids: Mutex::new(HashMap::new()),
            mvcc_col_updated_ids: Mutex::new(HashMap::new()),
        }
    }

    /// Idempotently seed the MVCC store with all existing records from amorphic storage
    /// for the specified tables.
    pub fn bootstrap_from_amorphic(&self, tables: &[String]) -> QueryResult<usize> {
        let mut total = 0usize;
        for table in tables {
            total += self.ensure_all_table_records_seeded(table)?;
        }
        Ok(total)
    }

    /// Ensure a single `(amorphic_record_id, row)` is present in `MvccStore` at `origin_ts`,
    /// returning `(mvcc_record_id, was_newly_inserted)`.
    fn ensure_single_record_seeded(
        &self,
        table: &str,
        amorphic_record_id: u64,
        row: &RowData,
        origin_ts: u64,
    ) -> QueryResult<(u64, bool)> {
        {
            let map = crate::lock_util::mutex_lock(&self.table_id_map);
            if let Some(&existing_mvcc_id) = map.get(table).and_then(|m| m.get(&amorphic_record_id))
            {
                return Ok((existing_mvcc_id, false));
            }
        }

        let mvcc_id = amorphic_record_id;
        let key = encode_record_key(table, mvcc_id);
        let mvcc_record = MvccRecord {
            columns: row.columns.clone(),
            values: row.values.clone(),
            amorphic_id: Some(amorphic_record_id),
        };
        let value = serde_json::to_vec(&mvcc_record).map_err(|e| {
            QueryError::ExecutionError(format!("Serialize error during MVCC seed: {}", e))
        })?;

        let mut map = crate::lock_util::mutex_lock(&self.table_id_map);
        let table_entry = map.entry(table.to_string()).or_default();
        if let Some(&existing_mvcc_id) = table_entry.get(&amorphic_record_id) {
            return Ok((existing_mvcc_id, false));
        }

        let inserted = self
            .tx_manager
            .store()
            .insert_committed_if_absent(&key, value, origin_ts)
            .map_err(|e| QueryError::ExecutionError(format!("MVCC seed error: {}", e)))?;
        table_entry.insert(amorphic_record_id, mvcc_id);
        Ok((mvcc_id, inserted))
    }

    /// Ensure all rows currently stored in `AmorphicTableStorage` for `table` have
    /// corresponding chains in `MvccStore` (used for full-table scans and non-indexed predicates).
    fn ensure_all_table_records_seeded(&self, table: &str) -> QueryResult<usize> {
        let rows_with_ids = match self.amorphic.scan_with_ids(table) {
            Ok(r) => r,
            Err(_) => return Ok(0),
        };
        let mut newly_seeded = 0usize;
        for (amorphic_id, row) in &rows_with_ids {
            let (_, added) = self.ensure_single_record_seeded(table, *amorphic_id, row, 0)?;
            if added {
                newly_seeded += 1;
            }
        }
        Ok(newly_seeded)
    }

    /// Synchronize any already-seeded MVCC chains for `table` after an external
    /// auto-commit `UPDATE`, `DELETE`, `TRUNCATE`, or `DROP TABLE` modified `AmorphicTableStorage`.
    /// O(1) no-op if `table` has no seeded MVCC chains.
    pub fn on_external_table_change(&self, table: &str) {
        let seeded_pairs: Vec<(u64, u64)> = {
            let map = crate::lock_util::mutex_lock(&self.table_id_map);
            match map.get(table) {
                Some(m) if !m.is_empty() => m.iter().map(|(&a, &m_id)| (a, m_id)).collect(),
                _ => return,
            }
        };

        let table_still_exists = self.amorphic.table_exists(table).unwrap_or(false);
        let commit_ts = advance_mvcc_timestamp();
        let store = self.tx_manager.store();
        let mut removed_amorphic_ids = Vec::new();

        for (amorphic_id, mvcc_id) in seeded_pairs {
            let key = encode_record_key(table, mvcc_id);
            let current_row = if table_still_exists {
                self.amorphic
                    .get_table_row_by_id(table, amorphic_id)
                    .ok()
                    .flatten()
            } else {
                None
            };

            match current_row {
                Some(row) => {
                    let changed = self
                        .get_committed_value(&key)
                        .and_then(|bytes| serde_json::from_slice::<MvccRecord>(&bytes).ok())
                        .map(|prev| prev.values != row.values || prev.columns != row.columns)
                        .unwrap_or(true);
                    if changed {
                        let updated_rec = MvccRecord {
                            columns: row.columns,
                            values: row.values,
                            amorphic_id: Some(amorphic_id),
                        };
                        if let Ok(bytes) = serde_json::to_vec(&updated_rec) {
                            let _ = store.append_external_committed(&key, Some(bytes), commit_ts);
                        }
                    }
                }
                None => {
                    let _ = store.append_external_committed(&key, None, commit_ts);
                    removed_amorphic_ids.push((amorphic_id, mvcc_id));
                }
            }
        }

        if !removed_amorphic_ids.is_empty() {
            let mut map = crate::lock_util::mutex_lock(&self.table_id_map);
            if !table_still_exists {
                map.remove(table);
            } else if let Some(tmap) = map.get_mut(table) {
                for (aid, _) in &removed_amorphic_ids {
                    tmap.remove(aid);
                }
            }
            let mut del = crate::lock_util::mutex_lock(&self.mvcc_deleted_ids);
            let del_set = del.entry(table.to_string()).or_default();
            for (_, mid) in removed_amorphic_ids {
                del_set.insert(mid);
            }
        }
    }

    /// Resolve visible `(mvcc_key, MvccRecord)` entries for `predicate` inside `session_id`:
    /// - Uses `O(k)` secondary index lookup + point `tx.get(&key)` when `predicate` has a
    ///   type-safe `col = literal` equality conjunct (avoiding 100k-row scans & false SSI
    ///   prefix conflicts!).
    /// - Falls back to `ensure_all_table_records_seeded` + `tx.scan_prefix` otherwise.
    fn resolve_session_candidates(
        &self,
        session_id: &str,
        table: &str,
        predicate: Option<&Expression>,
    ) -> QueryResult<Vec<(Vec<u8>, MvccRecord)>> {
        if let Some(pred) = predicate {
            if let Some((eq_col, eq_val)) =
                crate::amorphic_adapter::extract_conjunct_eq_predicate(pred)
            {
                if let Some(indexed_rows) =
                    self.amorphic.lookup_records_by_eq(table, &eq_col, &eq_val)?
                {
                    let mut candidate_mvcc_ids: Vec<u64> = Vec::with_capacity(indexed_rows.len() + 4);
                    let mut seen_ids: HashSet<u64> = HashSet::with_capacity(indexed_rows.len() + 4);

                    for (amorphic_id, row) in &indexed_rows {
                        let (mvcc_id, _) =
                            self.ensure_single_record_seeded(table, *amorphic_id, row, 0)?;
                        if seen_ids.insert(mvcc_id) {
                            candidate_mvcc_ids.push(mvcc_id);
                        }
                    }

                    // Include any rows deleted via MVCC or whose `eq_col` was modified via MVCC
                    {
                        let del = crate::lock_util::mutex_lock(&self.mvcc_deleted_ids);
                        if let Some(ids) = del.get(table) {
                            for &id in ids {
                                if seen_ids.insert(id) {
                                    candidate_mvcc_ids.push(id);
                                }
                            }
                        }
                    }
                    {
                        let upd = crate::lock_util::mutex_lock(&self.mvcc_col_updated_ids);
                        if let Some(ids) = upd.get(&(table.to_string(), eq_col)) {
                            for &id in ids {
                                if seen_ids.insert(id) {
                                    candidate_mvcc_ids.push(id);
                                }
                            }
                        }
                    }

                    let prefix = encode_table_prefix(table);
                    return self.with_session(session_id, |tx| {
                        // Also include any uncommitted keys for this table in `tx.write_set`
                        let local_keys: Vec<Vec<u8>> = tx
                            .write_set()
                            .keys()
                            .filter(|k| k.starts_with(&prefix))
                            .cloned()
                            .collect();
                        for lk in local_keys {
                            if let Some(mid) = decode_record_id(&lk, table) {
                                if seen_ids.insert(mid) {
                                    candidate_mvcc_ids.push(mid);
                                }
                            }
                        }

                        let mut out = Vec::with_capacity(candidate_mvcc_ids.len());
                        for mid in candidate_mvcc_ids {
                            let key = encode_record_key(table, mid);
                            if let Some(data) = tx.get(&key).map_err(|e| {
                                QueryError::ExecutionError(format!("MVCC get error: {}", e))
                            })? {
                                let record: MvccRecord =
                                    serde_json::from_slice(&data).map_err(|e| {
                                        QueryError::ExecutionError(format!(
                                            "Deserialize error: {}",
                                            e
                                        ))
                                    })?;
                                out.push((key, record));
                            }
                        }
                        Ok(out)
                    });
                }
            }
        }

        // General path: seed full table if needed, then scan_prefix
        self.ensure_all_table_records_seeded(table)?;
        let prefix = encode_table_prefix(table);
        let entries = self.with_session(session_id, |tx| {
            tx.scan_prefix(&prefix)
                .map_err(|e| QueryError::ExecutionError(format!("MVCC scan error: {}", e)))
        })?;
        let mut out = Vec::with_capacity(entries.len());
        for (key, data) in entries {
            let record: MvccRecord = serde_json::from_slice(&data)
                .map_err(|e| QueryError::ExecutionError(format!("Deserialize error: {}", e)))?;
            out.push((key, record));
        }
        Ok(out)
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

    /// Commit a transaction:
    /// 1. Validates SI/SSI conflicts under `commit_lock` BEFORE touching `AmorphicTableStorage`.
    /// 2. Persists all inserts/updates/deletes atomically under a single `AmorphicTableStorage`
    ///    write lock (`apply_atomic_batch_deferred`) so concurrent sessions never collide on
    ///    `"Batch already active"`.
    /// 3. Records newly assigned `amorphic_id`s on inserted MVCC records and in `table_id_map`,
    ///    publishes `commit_ts`, drops `commit_lock`, and waits on shared `GroupSync` `fsync`.
    pub fn commit_transaction(&self, session_id: &str) -> QueryResult<()> {
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

        if tx.write_set().is_empty() {
            tx.commit()
                .map_err(|e| QueryError::ExecutionError(format!("MVCC commit failed: {}", e)))?;
            return Ok(());
        }

        let mut pending_sync: Option<DeferredBatchSync> = None;

        tx.commit_with_hook(|write_set| {
            let mut ordered_keys: Vec<Vec<u8>> = Vec::with_capacity(write_set.len());
            let mut parsed_inserts: Vec<(usize, String, u64, MvccRecord)> = Vec::new();
            let mut deleted_tracking: Vec<(String, u64, Option<u64>)> = Vec::new();
            let mut updated_col_tracking: Vec<(String, String, u64)> = Vec::new();
            let mut batch_ops: Vec<AtomicBatchOp> = Vec::with_capacity(write_set.len());

            for (key, value) in write_set {
                let Some(null_pos) = key.iter().position(|&b| b == 0) else {
                    continue;
                };
                let table = std::str::from_utf8(&key[..null_pos]).unwrap_or("");
                if table.is_empty() {
                    continue;
                }
                let Some(mvcc_id) = decode_record_id(key, table) else {
                    continue;
                };

                match value {
                    Some(data) => {
                        let record = serde_json::from_slice::<MvccRecord>(data)
                            .map_err(|e| format!("Corrupt MVCC record on commit: {e}"))?;
                        let row = record.to_row_data();
                        if let Some(existing_id) = record.amorphic_id {
                            // Detect which columns actually changed relative to prior committed value
                            if let Some(prev_bytes) = self.get_committed_value(key) {
                                if let Ok(prev_rec) =
                                    serde_json::from_slice::<MvccRecord>(&prev_bytes)
                                {
                                    for ((col, new_v), old_v) in row
                                        .columns
                                        .iter()
                                        .zip(row.values.iter())
                                        .zip(prev_rec.values.iter())
                                    {
                                        if new_v != old_v {
                                            updated_col_tracking.push((
                                                table.to_string(),
                                                col.clone(),
                                                mvcc_id,
                                            ));
                                        }
                                    }
                                }
                            }
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
                            batch_ops.push(AtomicBatchOp::Update {
                                id: existing_id,
                                updates,
                            });
                        } else {
                            let key_idx = ordered_keys.len();
                            ordered_keys.push(key.clone());
                            parsed_inserts.push((key_idx, table.to_string(), mvcc_id, record));
                            batch_ops.push(AtomicBatchOp::Insert {
                                key_idx,
                                table: table.to_string(),
                                row,
                            });
                        }
                    }
                    None => {
                        let amorphic_id = self
                            .get_committed_value(key)
                            .and_then(|orig| serde_json::from_slice::<MvccRecord>(&orig).ok())
                            .and_then(|r| r.amorphic_id);
                        if let Some(aid) = amorphic_id {
                            batch_ops.push(AtomicBatchOp::Delete { id: aid });
                        }
                        deleted_tracking.push((table.to_string(), mvcc_id, amorphic_id));
                    }
                }
            }

            let (assigned_ids, sync_handle) = self
                .amorphic
                .apply_atomic_batch_deferred(batch_ops)
                .map_err(|e| format!("WAL batch error: {e}"))?;
            pending_sync = Some(sync_handle);

            // Rewrite newly inserted MVCC records so `amorphic_id` is populated before publish!
            let mut rewritten: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(assigned_ids.len());
            if !assigned_ids.is_empty() {
                let mut id_lookup: HashMap<usize, u64> = HashMap::with_capacity(assigned_ids.len());
                for (k_idx, real_id) in assigned_ids {
                    id_lookup.insert(k_idx, real_id);
                }
                let mut map = crate::lock_util::mutex_lock(&self.table_id_map);
                for (k_idx, tbl, mvcc_id, mut rec) in parsed_inserts {
                    if let Some(&real_id) = id_lookup.get(&k_idx) {
                        rec.amorphic_id = Some(real_id);
                        map.entry(tbl).or_default().insert(real_id, mvcc_id);
                        if let Ok(bytes) = serde_json::to_vec(&rec) {
                            rewritten.push((ordered_keys[k_idx].clone(), bytes));
                        }
                    }
                }
            }

            if !deleted_tracking.is_empty() {
                let mut map = crate::lock_util::mutex_lock(&self.table_id_map);
                let mut del = crate::lock_util::mutex_lock(&self.mvcc_deleted_ids);
                for (tbl, mid, aid_opt) in deleted_tracking {
                    if let Some(aid) = aid_opt {
                        if let Some(tmap) = map.get_mut(&tbl) {
                            tmap.remove(&aid);
                        }
                    }
                    del.entry(tbl).or_default().insert(mid);
                }
            }

            if !updated_col_tracking.is_empty() {
                let mut upd = crate::lock_util::mutex_lock(&self.mvcc_col_updated_ids);
                for (tbl, col, mid) in updated_col_tracking {
                    upd.entry((tbl, col)).or_default().insert(mid);
                }
            }

            Ok(rewritten)
        })
        .map_err(|e| QueryError::ExecutionError(format!("MVCC commit failed: {}", e)))?;

        // Wait for GroupSync fsync outside `commit_lock` so concurrent committers coalesce!
        if let Some(sync_handle) = pending_sync {
            sync_handle.wait_durable()?;
        }

        Ok(())
    }

    /// Rollback a transaction, discarding all buffered writes.
    /// Treats an already-aborted transaction (e.g. after a statement-level conflict) as `Ok(())`
    /// to match PostgreSQL `ROLLBACK` semantics.
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

        match tx.rollback() {
            Ok(()) | Err(joule_db_core::error::TransactionError::AlreadyAborted { .. }) => Ok(()),
            Err(e) => Err(QueryError::ExecutionError(format!(
                "MVCC rollback failed: {}",
                e
            ))),
        }
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
            Ok(())
        }
    }

    /// Rollback to a savepoint — restore the write_set from snapshot.
    pub fn rollback_to_savepoint(&self, session_id: &str, name: &str) -> Result<(), String> {
        let sessions = crate::lock_util::mutex_lock(&self.sessions);
        if let Some(session_mutex) = sessions.get(session_id) {
            let mut session = crate::lock_util::mutex_lock(session_mutex);
            let idx = session
                .savepoints
                .iter()
                .rposition(|sp| sp.name == name)
                .ok_or_else(|| format!("SAVEPOINT '{}' not found", name))?;
            let snapshot = session.savepoints[idx].clone();
            session.savepoints.truncate(idx + 1);
            if let Some(ref mut tx) = session.tx {
                tx.restore_write_set(snapshot.write_set_snapshot);
            }
            Ok(())
        } else {
            Ok(())
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
            Some(sid) => {
                self.ensure_all_table_records_seeded(table)?;
                self.with_session(sid, |tx| {
                    let prefix = encode_table_prefix(table);
                    let entries = tx.scan_prefix(&prefix).map_err(|e| {
                        QueryError::ExecutionError(format!("MVCC scan error: {}", e))
                    })?;
                    let mut rows = Vec::with_capacity(entries.len());
                    for (_key, data) in entries {
                        let record: MvccRecord = serde_json::from_slice(&data).map_err(|e| {
                            QueryError::ExecutionError(format!("Deserialize error: {}", e))
                        })?;
                        rows.push(record.to_row_data());
                    }
                    Ok(rows)
                })
            }
            None => self.amorphic.scan(table),
        }
    }

    /// Time-travel scan: read rows visible at a specific MVCC timestamp.
    pub fn scan_at_timestamp(&self, table: &str, ts: u64) -> QueryResult<Vec<RowData>> {
        self.ensure_all_table_records_seeded(table)?;
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
                let record_id = self.next_temp_id.fetch_add(1, Ordering::Relaxed);
                let key = encode_record_key(table, record_id);
                let record = MvccRecord::from_row_data(row);
                let data = serde_json::to_vec(&record)
                    .map_err(|e| QueryError::ExecutionError(format!("Serialize error: {}", e)))?;
                tx.put(&key, &data)
                    .map_err(|e| QueryError::ExecutionError(format!("MVCC put error: {}", e)))?;
                Ok(())
            }),
            None => {
                let amorphic_id = self.amorphic.insert_returning_id(table, row)?;
                // If an explicit MVCC session is concurrently active, stamp this auto-commit
                // row at the current MVCC timestamp so older snapshots do not see it prematurely.
                if self.active_session_count() > 0 {
                    let ts = advance_mvcc_timestamp();
                    let _ = self.ensure_single_record_seeded(table, amorphic_id, row, ts);
                }
                Ok(())
            }
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
                let candidates = self.resolve_session_candidates(sid, table, predicate)?;
                let mut updated = 0;
                for (key, record) in candidates {
                    let row = record.to_row_data();
                    let matches = predicate
                        .map(|p| evaluate_mvcc_predicate(&row, p))
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
            None => {
                let cnt = self.amorphic.update(table, assignments, predicate)?;
                if cnt > 0 {
                    self.on_external_table_change(table);
                }
                Ok(cnt)
            }
        }
    }

    /// Update matching rows by evaluating SET expressions per-row.
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
                let candidates = self.resolve_session_candidates(sid, table, predicate)?;
                let mut updated = 0;
                for (key, record) in candidates {
                    let row = record.to_row_data();
                    let matches = predicate
                        .map(|p| evaluate_mvcc_predicate(&row, p))
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
            None => {
                let cnt = self
                    .amorphic
                    .update_with_expressions(table, assignments, predicate, params)?;
                if cnt > 0 {
                    self.on_external_table_change(table);
                }
                Ok(cnt)
            }
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
                let candidates = self.resolve_session_candidates(sid, table, predicate)?;
                let mut deleted = 0;
                for (key, record) in candidates {
                    let row = record.to_row_data();
                    let matches = predicate
                        .map(|p| evaluate_mvcc_predicate(&row, p))
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
            None => {
                let cnt = self.amorphic.delete(table, predicate)?;
                if cnt > 0 {
                    self.on_external_table_change(table);
                }
                Ok(cnt)
            }
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
                let candidates = self.resolve_session_candidates(sid, table, predicate)?;
                let mut updated = 0;
                let mut returned_rows = Vec::new();
                for (key, record) in candidates {
                    let row = record.to_row_data();
                    let matches = predicate
                        .map(|p| evaluate_mvcc_predicate(&row, p))
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
            None => {
                let res = self.amorphic.update_with_expressions_returning(
                    table,
                    assignments,
                    predicate,
                    params,
                )?;
                if res.0 > 0 {
                    self.on_external_table_change(table);
                }
                Ok(res)
            }
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
                let candidates = self.resolve_session_candidates(sid, table, predicate)?;
                let mut deleted = 0;
                let mut returned_rows = Vec::new();
                for (key, record) in candidates {
                    let row = record.to_row_data();
                    let matches = predicate
                        .map(|p| evaluate_mvcc_predicate(&row, p))
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
            None => {
                let res = self.amorphic.delete_returning(table, predicate)?;
                if res.0 > 0 {
                    self.on_external_table_change(table);
                }
                Ok(res)
            }
        }
    }

    // ==================== Session reaper ====================

    /// Clean up sessions that have been idle longer than `timeout`.
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
        let mut session = crate::lock_util::mutex_lock(session_mutex);
        let tx = session.tx.as_mut().ok_or_else(|| {
            QueryError::ExecutionError("Transaction already consumed".to_string())
        })?;
        f(tx)
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
        let (_dir, _amorphic, mvcc) = setup_with_table("users", &["name", "age"]);

        let row = RowData::new(
            vec!["name".into(), "age".into()],
            vec![AstValue::String("Alice".into()), AstValue::Int(30)],
        );
        mvcc.insert(None, "users", &row).unwrap();

        let rows = mvcc.scan(None, "users").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("name"), Some(&AstValue::String("Alice".into())));
    }

    #[test]
    fn test_begin_commit() {
        let (_dir, amorphic, mvcc) = setup_with_table("users", &["name", "age"]);

        let sid = mvcc.begin_transaction();

        let row = RowData::new(
            vec!["name".into(), "age".into()],
            vec![AstValue::String("Bob".into()), AstValue::Int(25)],
        );
        mvcc.insert(Some(&sid), "users", &row).unwrap();

        let rows = mvcc.scan(Some(&sid), "users").unwrap();
        assert_eq!(rows.len(), 1);

        let amorphic_rows = amorphic.scan("users").unwrap();
        assert_eq!(amorphic_rows.len(), 0);

        mvcc.commit_transaction(&sid).unwrap();

        let amorphic_rows = amorphic.scan("users").unwrap();
        assert_eq!(amorphic_rows.len(), 1);
    }

    #[test]
    fn test_begin_rollback() {
        let (_dir, _amorphic, mvcc) = setup_with_table("users", &["name"]);

        let sid = mvcc.begin_transaction();

        let row = RowData::new(vec!["name".into()], vec![AstValue::String("Temp".into())]);
        mvcc.insert(Some(&sid), "users", &row).unwrap();

        let rows = mvcc.scan(Some(&sid), "users").unwrap();
        assert_eq!(rows.len(), 1);

        mvcc.rollback_transaction(&sid).unwrap();

        assert!(mvcc.scan(Some(&sid), "users").is_err());
    }

    #[test]
    fn test_snapshot_isolation() {
        let (_dir, _amorphic, mvcc) = setup_with_table("t", &["val"]);

        let row = RowData::new(vec!["val".into()], vec![AstValue::Int(1)]);
        mvcc.insert(None, "t", &row).unwrap();

        let sid_a = mvcc.begin_transaction();
        mvcc.bootstrap_from_amorphic(&["t".to_string()]).unwrap();
        let sid_b = mvcc.begin_transaction();

        let row2 = RowData::new(vec!["val".into()], vec![AstValue::Int(2)]);
        mvcc.insert(Some(&sid_a), "t", &row2).unwrap();
        mvcc.commit_transaction(&sid_a).unwrap();

        let rows_b = mvcc.scan(Some(&sid_b), "t").unwrap();
        assert_eq!(rows_b.len(), 1);
        assert_eq!(rows_b[0].get("val"), Some(&AstValue::Int(1)));

        mvcc.rollback_transaction(&sid_b).unwrap();
    }

    #[test]
    fn test_write_write_conflict() {
        let (_dir, _amorphic, mvcc) = setup_with_table("t", &["val"]);

        mvcc.bootstrap_from_amorphic(&["t".to_string()]).ok();

        let sid_a = mvcc.begin_transaction();
        let sid_b = mvcc.begin_transaction();

        let row = RowData::new(vec!["val".into()], vec![AstValue::Int(1)]);
        mvcc.insert(Some(&sid_a), "t", &row).unwrap();

        let row2 = RowData::new(vec!["val".into()], vec![AstValue::Int(2)]);
        mvcc.insert(Some(&sid_b), "t", &row2).unwrap();

        mvcc.commit_transaction(&sid_a).unwrap();
        mvcc.commit_transaction(&sid_b).unwrap();
    }

    #[test]
    fn test_bootstrap_from_amorphic() {
        let (_dir, amorphic, mvcc) = setup_with_table("users", &["name", "age"]);

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

        let count = mvcc
            .bootstrap_from_amorphic(&["users".to_string()])
            .unwrap();
        assert_eq!(count, 2);
        // Second call is idempotent (0 duplicates)
        assert_eq!(
            mvcc.bootstrap_from_amorphic(&["users".to_string()])
                .unwrap(),
            0
        );

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

    fn reopen_from_path(path: &std::path::Path) -> (Arc<AmorphicTableStorage>, MvccTableStorage) {
        let store = DurableAmorphicStore::open(path).unwrap();
        let amorphic = Arc::new(AmorphicTableStorage::new(store));
        let mvcc = MvccTableStorage::new(amorphic.clone());
        (amorphic, mvcc)
    }

    #[test]
    fn test_explicit_tx_commit_durable() {
        let dir = TempDir::new().unwrap();

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

        {
            let (amorphic, _mvcc) = reopen_from_path(dir.path());
            let rows = amorphic.scan("users").unwrap();
            assert_eq!(rows.len(), 0);
        }
    }

    #[test]
    fn test_crash_before_commit_loses_data() {
        let dir = TempDir::new().unwrap();

        {
            let store = DurableAmorphicStore::open(dir.path()).unwrap();
            let amorphic = Arc::new(AmorphicTableStorage::new(store));
            let mvcc = MvccTableStorage::new(amorphic.clone());

            amorphic.create_table("users", &["name".into()]).unwrap();

            let sid = mvcc.begin_transaction();
            let row = RowData::new(vec!["name".into()], vec![AstValue::String("Lost".into())]);
            mvcc.insert(Some(&sid), "users", &row).unwrap();
        }

        {
            let (amorphic, _mvcc) = reopen_from_path(dir.path());
            let rows = amorphic.scan("users").unwrap();
            assert_eq!(rows.len(), 0);
        }
    }

    #[test]
    fn test_multi_row_transaction_atomicity() {
        let dir = TempDir::new().unwrap();

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

        {
            let (amorphic, _mvcc) = reopen_from_path(dir.path());
            let rows = amorphic.scan("items").unwrap();
            assert_eq!(rows.len(), 3);
        }
    }

    #[test]
    fn test_delete_in_transaction_durable() {
        let dir = TempDir::new().unwrap();

        {
            let store = DurableAmorphicStore::open(dir.path()).unwrap();
            let amorphic = Arc::new(AmorphicTableStorage::new(store));
            let mvcc = MvccTableStorage::new(amorphic.clone());

            amorphic.create_table("t", &["name".into()]).unwrap();

            let row = RowData::new(
                vec!["name".into()],
                vec![AstValue::String("ToDelete".into())],
            );
            mvcc.insert(None, "t", &row).unwrap();

            // Works even WITHOUT calling `bootstrap_from_amorphic` manually!
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

        {
            let (amorphic, _mvcc) = reopen_from_path(dir.path());
            let rows = amorphic.scan("t").unwrap();
            assert_eq!(rows.len(), 0, "Deleted record should not survive restart");
        }
    }

    #[test]
    fn test_update_in_transaction_durable() {
        let dir = TempDir::new().unwrap();

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

        {
            let store = DurableAmorphicStore::open(dir.path()).unwrap();
            let amorphic = Arc::new(AmorphicTableStorage::new(store));
            let mvcc = MvccTableStorage::new(amorphic.clone());

            amorphic.create_table("t", &["val".into()]).unwrap();

            let row = RowData::new(vec!["val".into()], vec![AstValue::Int(99)]);
            mvcc.insert(None, "t", &row).unwrap();
        }

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

        let sid = mvcc.begin_transaction();
        let row = RowData::new(
            vec!["name".into(), "val".into()],
            vec![AstValue::String("a".into()), AstValue::Int(1)],
        );
        mvcc.insert(Some(&sid), "t", &row).unwrap();

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

        let rows = mvcc.scan(Some(&sid), "t").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("val"), Some(&AstValue::Int(42)));

        mvcc.commit_transaction(&sid).unwrap();
    }

    #[test]
    fn test_delete_in_transaction() {
        let (_dir, _amorphic, mvcc) = setup_with_table("t", &["name"]);

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

        let pred = Expression::Binary {
            left: Box::new(Expression::Column("name".into())),
            op: joule_db_query::ast::Operator::Eq,
            right: Box::new(Expression::Literal(AstValue::String("remove".into()))),
        };
        let deleted = mvcc.delete(Some(&sid), "t", Some(&pred)).unwrap();
        assert_eq!(deleted, 1);

        let rows = mvcc.scan(Some(&sid), "t").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("name"), Some(&AstValue::String("keep".into())));

        mvcc.commit_transaction(&sid).unwrap();
    }

    #[test]
    fn test_insert_in_tx_then_update_in_later_tx_no_duplicate_row() {
        let (_dir, amorphic, mvcc) = setup_with_table("accounts", &["id", "balance"]);

        // TX 1 inserts row (id=1, balance=100)
        let sid1 = mvcc.begin_transaction();
        mvcc.insert(
            Some(&sid1),
            "accounts",
            &RowData::new(
                vec!["id".into(), "balance".into()],
                vec![AstValue::Int(1), AstValue::Int(100)],
            ),
        )
        .unwrap();
        mvcc.commit_transaction(&sid1).unwrap();
        assert_eq!(amorphic.scan("accounts").unwrap().len(), 1);

        // TX 2 updates row (id=1) -> balance=250; must NOT create a duplicate row!
        let sid2 = mvcc.begin_transaction();
        let mut assignments = HashMap::new();
        assignments.insert("balance".to_string(), AstValue::Int(250));
        let pred = Expression::Binary {
            left: Box::new(Expression::Column("id".into())),
            op: joule_db_query::ast::Operator::Eq,
            right: Box::new(Expression::Literal(AstValue::Int(1))),
        };
        assert_eq!(
            mvcc.update(Some(&sid2), "accounts", &assignments, Some(&pred))
                .unwrap(),
            1
        );
        mvcc.commit_transaction(&sid2).unwrap();

        let rows = amorphic.scan("accounts").unwrap();
        assert_eq!(rows.len(), 1, "UPDATE after MVCC INSERT must not duplicate row");
        assert_eq!(rows[0].get("balance"), Some(&AstValue::Int(250)));
    }

    #[test]
    fn test_concurrent_multithread_commit_atomic_batch_conservation() {
        use crate::query::{QueryExecutor, QueryRequest, SimpleQueryExecutor};

        let dir = TempDir::new().unwrap();
        let store = DurableAmorphicStore::open(dir.path()).unwrap();
        let amorphic = Arc::new(AmorphicTableStorage::new(store));
        let exec = Arc::new(SimpleQueryExecutor::with_amorphic(amorphic));

        let run_sql = |e: &SimpleQueryExecutor, sql: &str, sid: Option<String>| {
            e.execute(&QueryRequest {
                sql: sql.to_string(),
                session_id: sid,
                ..Default::default()
            })
        };

        run_sql(
            &exec,
            "CREATE TABLE accounts (id BIGINT, branch_id BIGINT, balance BIGINT)",
            None,
        )
        .unwrap();
        run_sql(
            &exec,
            "CREATE TABLE history (tx_seq BIGINT, acct_id BIGINT, delta BIGINT)",
            None,
        )
        .unwrap();

        // Populate 200 accounts via COPY / bulk_copy AFTER SimpleQueryExecutor startup!
        let copy_rows: Vec<Vec<Option<String>>> = (1..=200)
            .map(|i| {
                vec![
                    Some(i.to_string()),
                    Some(((i - 1) / 20 + 1).to_string()),
                    Some("1000".to_string()),
                ]
            })
            .collect();
        let copied = exec
            .bulk_copy(
                "accounts",
                &["id".into(), "branch_id".into(), "balance".into()],
                &copy_rows,
            )
            .unwrap()
            .unwrap();
        assert_eq!(copied, 200);

        // Spawn 8 concurrent workers updating disjoint account ranges via BEGIN..UPDATE..INSERT..COMMIT
        let mut handles = Vec::new();
        for w in 0..8i64 {
            let ex = exec.clone();
            handles.push(std::thread::spawn(move || {
                let mut local_delta = 0i64;
                for step in 0..15i64 {
                    let aid = w * 25 + (step % 25) + 1;
                    let delta = if step % 2 == 0 { 50i64 } else { -20i64 };
                    let begin_resp = ex
                        .execute(&QueryRequest {
                            sql: "BEGIN".into(),
                            ..Default::default()
                        })
                        .unwrap();
                    let sid = begin_resp.session_id.clone().unwrap();
                    let upd = ex
                        .execute(&QueryRequest {
                            sql: format!(
                                "UPDATE accounts SET balance = balance + ({delta}) WHERE id = {aid}"
                            ),
                            session_id: Some(sid.clone()),
                            ..Default::default()
                        })
                        .unwrap();
                    assert_eq!(upd.affected_rows, Some(1));
                    ex.execute(&QueryRequest {
                        sql: format!(
                            "INSERT INTO history (tx_seq, acct_id, delta) VALUES ({}, {aid}, {delta})",
                            w * 100 + step
                        ),
                        session_id: Some(sid.clone()),
                        ..Default::default()
                    })
                    .unwrap();
                    ex.execute(&QueryRequest {
                        sql: "COMMIT".into(),
                        session_id: Some(sid),
                        ..Default::default()
                    })
                    .unwrap();
                    local_delta += delta;
                }
                local_delta
            }));
        }

        let total_delta: i64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        let check = run_sql(&exec, "SELECT COUNT(*), SUM(balance) FROM accounts", None).unwrap();
        assert_eq!(check.rows[0][0], serde_json::json!(200));
        assert_eq!(
            check.rows[0][1],
            serde_json::json!(200 * 1000 + total_delta),
            "Exact balance conservation must hold across concurrent MVCC updates after COPY"
        );
        let hist = run_sql(&exec, "SELECT COUNT(*), SUM(delta) FROM history", None).unwrap();
        assert_eq!(hist.rows[0][0], serde_json::json!(8 * 15));
        assert_eq!(hist.rows[0][1], serde_json::json!(total_delta));
    }
}
