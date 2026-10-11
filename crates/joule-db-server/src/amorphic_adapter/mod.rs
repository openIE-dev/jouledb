//! AmorphicTableStorage - bridges SQL queries to the Amorphic store
//!
//! Uses a `__table__` field convention to map SQL tables onto amorphic records.
//! Schema metadata is stored as amorphic records with `__table__ = "__schemas__"`.
//!
//! This lets the query executor treat `DurableAmorphicStore` as a collection
//! of SQL tables, while the amorphic store handles HDC encoding, indexing,
//! and graph edges transparently.

mod constraints;
mod conversions;
mod crud;
mod expressions;
mod fulltext;
mod indexes;
mod schema;
#[cfg(test)]
mod tests;
mod triggers;
mod views;

use std::collections::HashMap;
use std::sync::RwLock;

use joule_db_amorphic::{AmorphicRecord, DurableAmorphicStore, RecordId, Value as AmorphicValue};
use joule_db_query::ast::{Expression, Operator, UnaryOperator, Value as AstValue};
use joule_db_query::error::{QueryError, QueryResult};
use joule_db_query::executor::{RowData, TableStorage};
use joule_db_query::planner::{
    ColumnStats as PlannerColumnStats, IndexInfo, IndexType, Statistics as PlannerStatistics,
    TableStats as PlannerTableStats,
};
use joule_db_query::sql::SqlColumnDef;

// Re-export public items that external code references
pub use conversions::{ast_to_amorphic_value, ast_to_json, expression_to_sql};

/// Reserved field name that identifies which SQL table a record belongs to
pub(super) const TABLE_FIELD: &str = "__table__";
/// Reserved table name for sequence metadata (auto-increment counters)
pub(super) const SEQUENCE_TABLE: &str = "__sequences__";
/// Field for sequence name (table.column)
pub(super) const SEQUENCE_NAME_FIELD: &str = "__sequence_name__";
/// Field for current sequence value
pub(super) const SEQUENCE_VALUE_FIELD: &str = "__sequence_value__";

/// Reserved table name for schema metadata
pub(super) const SCHEMA_TABLE: &str = "__schemas__";
/// Field in schema records that stores the table name
pub(super) const SCHEMA_NAME_FIELD: &str = "__schema_name__";
/// Field in schema records that stores column definitions (JSON array)
pub(super) const SCHEMA_COLUMNS_FIELD: &str = "__columns__";
/// Reserved table name for index metadata
pub(super) const INDEX_TABLE: &str = "__indexes__";
/// Field in schema records that stores full column definitions (JSON array of objects)
pub(super) const SCHEMA_COLUMN_DEFS_FIELD: &str = "__column_defs__";
/// Field in index records that stores the index name
pub(super) const INDEX_NAME_FIELD: &str = "__index_name__";
/// Field in index records that stores the target table
pub(super) const INDEX_TABLE_REF: &str = "__index_table__";
/// Field in index records that stores indexed columns
pub(super) const INDEX_COLUMNS_FIELD: &str = "__index_columns__";
/// Field in index records that stores unique flag
pub(super) const INDEX_UNIQUE_FIELD: &str = "__index_unique__";
/// Reserved table name for trigger metadata
pub(super) const TRIGGER_TABLE: &str = "__triggers__";
/// Field in trigger records that stores the trigger name
pub(super) const TRIGGER_NAME_FIELD: &str = "__trigger_name__";
/// Field in trigger records that stores the target table
pub(super) const TRIGGER_TABLE_REF: &str = "__trigger_table__";
/// Field in trigger records that stores the timing (BEFORE/AFTER)
pub(super) const TRIGGER_TIMING_FIELD: &str = "__trigger_timing__";
/// Field in trigger records that stores the event (INSERT/UPDATE/DELETE)
pub(super) const TRIGGER_EVENT_FIELD: &str = "__trigger_event__";
/// Field in trigger records that stores the body SQL
pub(super) const TRIGGER_BODY_FIELD: &str = "__trigger_body__";
/// Field in schema records that stores column families
pub(super) const SCHEMA_COLUMN_FAMILIES_FIELD: &str = "__column_families__";
/// Reserved table name for materialized view metadata
pub(super) const MATVIEW_TABLE: &str = "__matviews__";
/// Field in matview records that stores the view name
pub(super) const MATVIEW_NAME_FIELD: &str = "__matview_name__";
/// Field in matview records that stores the query SQL
pub(super) const MATVIEW_QUERY_FIELD: &str = "__matview_query__";
/// Field in matview records that stores the backing table name
pub(super) const MATVIEW_BACKING_TABLE_FIELD: &str = "__matview_backing_table__";
/// Reserved table name for reference metadata
pub(super) const REFERENCE_TABLE: &str = "__references__";
/// Field in reference records that stores the reference name
pub(super) const REFERENCE_NAME_FIELD: &str = "__reference_name__";
/// Field in reference records that stores the source table
pub(super) const REFERENCE_SRC_TABLE_FIELD: &str = "__reference_src_table__";
/// Field in reference records that stores the source column
pub(super) const REFERENCE_SRC_COLUMN_FIELD: &str = "__reference_src_column__";
/// Field in reference records that stores the target table
pub(super) const REFERENCE_TGT_TABLE_FIELD: &str = "__reference_tgt_table__";
/// Field in reference records that stores the target column
pub(super) const REFERENCE_TGT_COLUMN_FIELD: &str = "__reference_tgt_column__";

/// Reserved table name for bucket metadata
pub(super) const BUCKET_TABLE: &str = "__buckets__";
/// Field in bucket records that stores the bucket name
pub(super) const BUCKET_NAME_FIELD: &str = "__bucket_name__";
/// Field in bucket records that stores optional max size in bytes
pub(super) const BUCKET_MAX_SIZE_FIELD: &str = "__bucket_max_size__";

/// Reserved table name for API endpoint metadata
pub(super) const API_TABLE: &str = "__apis__";
/// Field in API records that stores the endpoint path
pub(super) const API_PATH_FIELD: &str = "__api_path__";
/// Field in API records that stores the HTTP method
pub(super) const API_METHOD_FIELD: &str = "__api_method__";
/// Field in API records that stores the handler SQL
pub(super) const API_HANDLER_FIELD: &str = "__api_handler__";

/// Default collection name for schema-less ingest
pub(super) const DEFAULT_COLLECTION: &str = "__default__";
/// Field name for specifying collection in unified ingest
pub(super) const COLLECTION_FIELD: &str = "_collection";

/// Trigger info retrieved from metadata
#[derive(Debug, Clone)]
pub struct TriggerInfo {
    pub name: String,
    pub table: String,
    pub timing: String, // "BEFORE" or "AFTER"
    pub event: String,  // "INSERT", "UPDATE", or "DELETE"
    pub body: String,   // SQL statement to execute
}

/// Reference definition info retrieved from metadata
#[derive(Debug, Clone)]
pub struct ReferenceInfo {
    pub name: String,
    pub src_table: String,
    pub src_column: String,
    pub tgt_table: String,
    pub tgt_column: String,
}

/// Bucket definition info retrieved from metadata
#[derive(Debug, Clone)]
pub struct BucketInfo {
    pub name: String,
    pub max_size: Option<u64>,
}

/// API endpoint definition info retrieved from metadata
#[derive(Debug, Clone)]
pub struct ApiEndpointInfo {
    pub path: String,
    pub method: String,
    pub handler_sql: String,
}

/// Foreign key info stored in schema metadata
#[derive(Debug, Clone)]
pub struct ForeignKeyInfo {
    pub ref_table: String,
    pub ref_column: String,
    pub on_delete: Option<String>,
    pub on_update: Option<String>,
}

/// Column definition info retrieved from schema metadata
#[derive(Debug, Clone)]
pub struct ColumnDefInfo {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
    pub primary_key: bool,
    pub unique: bool,
    /// Default value as a JSON-serializable value (if any)
    pub default_value: Option<serde_json::Value>,
    /// CHECK constraint expression as serialized string (if any)
    pub check_expr: Option<String>,
    /// Whether this column auto-increments (SERIAL/BIGSERIAL)
    pub auto_increment: bool,
    /// Foreign key constraint (if any)
    pub foreign_key: Option<ForeignKeyInfo>,
    /// Column family assignment (for wide-column storage)
    pub column_family: Option<String>,
    /// Computed column expression as serialized SQL string (if any)
    pub computed_expr: Option<String>,
}

/// Collected statistics for a table
#[derive(Debug, Clone)]
pub struct TableStatistics {
    pub row_count: usize,
    pub columns: HashMap<String, ColumnStatistics>,
    /// (index_name, columns, unique, index_type)
    pub indexes: Vec<(String, Vec<String>, bool, String)>,
}

/// Statistics for a single column
#[derive(Debug, Clone)]
pub struct ColumnStatistics {
    pub distinct_count: usize,
    pub null_fraction: f64,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
}

/// Adapter that exposes a `DurableAmorphicStore` as SQL-queryable tables.
///
/// Each SQL table is represented as amorphic records sharing the same
/// `__table__` field value. Column schemas are stored as dedicated
/// `__schemas__` records for `CREATE TABLE` / `DROP TABLE` support.
/// The store lock, plus background catch-up of the hypervector work that
/// bulk ingest defers. `write()` is the plain write lock. `read()` and
/// `read_primary()` never do hypervector work; `read_indexed()` (similarity
/// search) finishes the queued work first. The background thread catches up
/// in bounded steps once writes go idle.
pub(crate) struct HoloLock {
    inner: std::sync::Arc<RwLock<DurableAmorphicStore>>,
    pending: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// When bulk ingest last queued work (ms since `epoch`): the background
    /// catch-up waits for a short write-idle gap so it doesn't take the
    /// lock from a streaming COPY.
    last_bulk_ms: std::sync::Arc<std::sync::atomic::AtomicU64>,
    epoch: std::time::Instant,
    /// Group commit (fsync per commit, shared by concurrent writers) unless
    /// JOULEDB_SYNC_COMMIT=off. None = relaxed (flush to OS only).
    sync: Option<std::sync::Arc<joule_db_local::storage::wal::GroupSync>>,
}

/// `JOULEDB_SYNC_COMMIT`: `on` (default) makes every commit power-loss
/// durable before it is acknowledged; `off` acknowledges once the WAL is in
/// the OS page cache (survives a process crash, not a power loss).
pub fn sync_commit_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("JOULEDB_SYNC_COMMIT").as_deref().map(str::to_ascii_lowercase).as_deref(),
            Ok("off") | Ok("0") | Ok("false")
        )
    })
}

/// Write guard that, when dropped, releases the store lock first and then
/// waits for its commits to be durable (group commit: writers queued behind
/// one fsync share it).
pub(crate) struct SyncWriteGuard<'a> {
    guard: Option<std::sync::RwLockWriteGuard<'a, DurableAmorphicStore>>,
    sync: Option<&'a joule_db_local::storage::wal::GroupSync>,
}

impl std::ops::Deref for SyncWriteGuard<'_> {
    type Target = DurableAmorphicStore;
    fn deref(&self) -> &DurableAmorphicStore {
        self.guard.as_ref().expect("guard held until drop")
    }
}

impl std::ops::DerefMut for SyncWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut DurableAmorphicStore {
        self.guard.as_mut().expect("guard held until drop")
    }
}

impl Drop for SyncWriteGuard<'_> {
    fn drop(&mut self) {
        let lsn = self.sync.map(|g| g.committed_lsn());
        drop(self.guard.take());
        if let (Some(g), Some(lsn)) = (self.sync, lsn) {
            if let Err(e) = g.wait_durable(lsn) {
                // sticky: GroupSync reports it on every later commit too
                tracing::error!("commit durability: {e}");
                SYNC_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

/// fsync failures seen by commit guards (metric; also sticky in GroupSync)
pub static SYNC_FAILURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl HoloLock {
    /// Records per background catch-up step (bounds how long it holds the lock).
    const STEP: usize = 2_048;
    /// Write-idle gap before the background catch-up runs.
    const IDLE_MS: u64 = 25;

    fn new(store: DurableAmorphicStore) -> Self {
        let sync = if sync_commit_enabled() {
            match store.group_sync() {
                Ok(g) => Some(std::sync::Arc::new(g)),
                Err(e) => {
                    tracing::warn!("group commit unavailable ({e}); commits flush to OS only");
                    None
                }
            }
        } else {
            None
        };
        let inner = std::sync::Arc::new(RwLock::new(store));
        let pending = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let weak = std::sync::Arc::downgrade(&inner);
        let flag = pending.clone();
        let last_bulk_ms = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let last = last_bulk_ms.clone();
        let epoch = std::time::Instant::now();
        let _ = std::thread::Builder::new()
            .name("jouledb-holo-catchup".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    let Some(lock) = weak.upgrade() else { return };
                    if !flag.load(std::sync::atomic::Ordering::Acquire) {
                        continue;
                    }
                    let now = epoch.elapsed().as_millis() as u64;
                    if now.saturating_sub(last.load(std::sync::atomic::Ordering::Relaxed))
                        < Self::IDLE_MS
                    {
                        continue;
                    }
                    // try_write: never queue behind a streaming bulk write and
                    // then grab the lock in the gap between its batches.
                    if let Ok(mut g) = lock.try_write() {
                        let now = epoch.elapsed().as_millis() as u64;
                        if now.saturating_sub(last.load(std::sync::atomic::Ordering::Relaxed))
                            < Self::IDLE_MS
                        {
                            continue;
                        }
                        g.catch_up_holograms(Self::STEP);
                        if g.pending_holograms() == 0 {
                            flag.store(false, std::sync::atomic::Ordering::Release);
                        }
                    }
                }
            });
        Self {
            inner,
            pending,
            last_bulk_ms,
            epoch,
            sync,
        }
    }

    pub(crate) fn write(&self) -> std::sync::LockResult<SyncWriteGuard<'_>> {
        let sync = self.sync.as_deref();
        match self.inner.write() {
            // after an fsync failure, refuse further writes (nothing more is
            // acknowledged on a WAL that may not be durable)
            Ok(g) if sync.is_some_and(|s| s.has_failed()) => Err(std::sync::PoisonError::new(
                SyncWriteGuard { guard: Some(g), sync: None },
            )),
            Ok(g) => Ok(SyncWriteGuard { guard: Some(g), sync }),
            Err(p) => Err(std::sync::PoisonError::new(SyncWriteGuard {
                guard: Some(p.into_inner()),
                sync,
            })),
        }
    }

    /// The group-commit handle (None when JOULEDB_SYNC_COMMIT=off).
    pub(crate) fn group_sync(&self) -> Option<&joule_db_local::storage::wal::GroupSync> {
        self.sync.as_deref()
    }

    /// Plain read lock. Primary data and the field/value/columnar indexes
    /// are always current; deferred hypervector work is left to the
    /// background thread so analytics never pay for it (a bounded per-read
    /// catch-up put ~7 ms on every query until a 1M-row backlog drained).
    pub(crate) fn read(
        &self,
    ) -> std::sync::LockResult<std::sync::RwLockReadGuard<'_, DurableAmorphicStore>> {
        self.inner.read()
    }

    /// Read lock for hypervector consumers (similarity search): finishes any
    /// queued hypervector work first so results cover every committed row.
    pub(crate) fn read_indexed(
        &self,
    ) -> std::sync::LockResult<std::sync::RwLockReadGuard<'_, DurableAmorphicStore>> {
        if self.pending.load(std::sync::atomic::Ordering::Acquire) {
            if let Ok(mut g) = self.inner.write() {
                g.catch_up_holograms(usize::MAX);
                self.pending
                    .store(false, std::sync::atomic::Ordering::Release);
            }
        }
        self.inner.read()
    }

    pub(crate) fn read_primary(
        &self,
    ) -> std::sync::LockResult<std::sync::RwLockReadGuard<'_, DurableAmorphicStore>> {
        self.inner.read()
    }

    /// Note that bulk ingest queued hypervector work.
    pub(crate) fn mark_pending(&self) {
        self.last_bulk_ms.store(
            self.epoch.elapsed().as_millis() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.pending
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

pub struct AmorphicTableStorage {
    store: HoloLock,
    /// Columnar snapshots per table, keyed by the store generation they were built at
    col_cache: std::sync::Mutex<
        HashMap<
            String,
            (
                (u64, usize),
                std::sync::Arc<crate::columnar_exec::TableColumns>,
            ),
        >,
    >,
}

impl AmorphicTableStorage {
    pub fn new(store: DurableAmorphicStore) -> Self {
        Self {
            store: HoloLock::new(store),
            col_cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Typed column vectors for `table`, rebuilt only when the store changed.
    /// `None` for views and unknown tables (callers use the general executor).
    pub fn table_columns(
        &self,
        table: &str,
    ) -> Option<std::sync::Arc<crate::columnar_exec::TableColumns>> {
        if self.get_view(table).is_some() {
            return None;
        }
        let store = self.store.read_primary().ok()?;
        let generation = store.generation();
        if let Some((g, t)) = self.col_cache.lock().ok()?.get(table) {
            if *g == generation {
                return Some(t.clone());
            }
        }
        let columns = self.get_schema_columns(&store, table).ok()?;
        if columns.is_empty() {
            return None;
        }
        if std::env::var_os("JOULEDB_VEC_TRACE").is_some() {
            let old = self
                .col_cache
                .lock()
                .ok()
                .and_then(|c| c.get(table).map(|(g, _)| *g));
            eprintln!(
                "colcache miss table={table} cached={old:?} now={generation:?} pending={}",
                store.pending_holograms()
            );
        }
        let recs = store.records_equal(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let t = std::sync::Arc::new(crate::columnar_exec::TableColumns::build(&columns, &recs));
        drop(store);
        let mut cache = self.col_cache.lock().ok()?;
        if cache.len() > 64 {
            cache.clear();
        }
        cache.insert(table.to_string(), (generation, t.clone()));
        Some(t)
    }

    /// After a bulk load committed `ids` into `table`, bring that table's
    /// column snapshot forward from the batch alone (no full rebuild), so the
    /// first query after a load is not the one that pays for the build.
    /// `before` is the store generation just before the batch.
    pub(crate) fn refresh_columns_after_load(
        &self,
        store: &DurableAmorphicStore,
        table: &str,
        before: (u64, usize),
        ids: &[u64],
    ) {
        let Ok(mut cache) = self.col_cache.lock() else {
            return;
        };
        let now = store.generation();
        match cache.get_mut(table) {
            Some((g, t)) if *g == before => {
                use rayon::prelude::*;
                let recs: Vec<&joule_db_amorphic::AmorphicRecord> =
                    ids.par_iter().filter_map(|&id| store.get(id)).collect();
                if recs.len() == ids.len() && std::sync::Arc::make_mut(t).append(&recs) {
                    *g = now;
                } else {
                    cache.remove(table);
                }
            }
            Some(_) => {
                cache.remove(table);
            }
            None => {
                // first load into this table: build once now, on the writer
                drop(cache);
                let Ok(columns) = self.get_schema_columns(store, table) else {
                    return;
                };
                if columns.is_empty() {
                    return;
                }
                let recs =
                    store.records_equal(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
                let t =
                    std::sync::Arc::new(crate::columnar_exec::TableColumns::build(&columns, &recs));
                if let Ok(mut cache) = self.col_cache.lock() {
                    if cache.len() > 64 {
                        cache.clear();
                    }
                    cache.insert(table.to_string(), (now, t));
                }
            }
        }
    }

    /// List all user-created table names (from schema metadata).
    pub fn list_tables(&self) -> Vec<String> {
        let store = match self.store.read() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let schema_result = store.query_equals(
            TABLE_FIELD,
            &AmorphicValue::String(SCHEMA_TABLE.to_string()),
        );
        let mut names = Vec::new();
        for record in schema_result.records() {
            if let Some(AmorphicValue::String(name)) = record.get(SCHEMA_NAME_FIELD) {
                names.push(name.clone());
            }
        }
        names
    }

    /// Convert an AmorphicRecord to a RowData given column names
    pub(super) fn record_to_row(record: &AmorphicRecord, columns: &[String]) -> RowData {
        let values: Vec<AstValue> = columns
            .iter()
            .map(|col| {
                record
                    .get(col)
                    .map(conversions::amorphic_to_ast)
                    .unwrap_or(AstValue::Null)
            })
            .collect();
        RowData::new(columns.to_vec(), values)
    }

    /// Get schema columns for a table
    pub(super) fn get_schema_columns(
        &self,
        store: &DurableAmorphicStore,
        table: &str,
    ) -> QueryResult<Vec<String>> {
        let result =
            store.query_equals(SCHEMA_NAME_FIELD, &AmorphicValue::String(table.to_string()));
        for record in result.records() {
            if record.get(TABLE_FIELD) == Some(&AmorphicValue::String(SCHEMA_TABLE.to_string())) {
                if let Some(AmorphicValue::Array(cols)) = record.get(SCHEMA_COLUMNS_FIELD) {
                    return Ok(cols
                        .iter()
                        .filter_map(|v| match v {
                            AmorphicValue::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect());
                }
            }
        }
        Err(QueryError::ExecutionError(format!(
            "Table '{}' does not exist",
            table
        )))
    }
}

// ============ TableStorage implementation ============

impl TableStorage for AmorphicTableStorage {
    fn scan(&self, table: &str) -> QueryResult<Vec<RowData>> {
        let store = self.store.read().map_err(conversions::lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        let rows = store
            .records_equal(TABLE_FIELD, &AmorphicValue::String(table.to_string()))
            .into_iter()
            .map(|r| Self::record_to_row(r, &columns))
            .collect();

        Ok(rows)
    }

    fn columns(&self, table: &str) -> QueryResult<Vec<String>> {
        let store = self.store.read().map_err(conversions::lock_error)?;
        self.get_schema_columns(&store, table)
    }

    fn insert(&self, table: &str, row: &RowData) -> QueryResult<()> {
        self.insert_returning_id(table, row)?;
        Ok(())
    }

    fn update(
        &self,
        table: &str,
        assignments: &HashMap<String, AstValue>,
        predicate: Option<&Expression>,
    ) -> QueryResult<usize> {
        let mut store = self.store.write().map_err(conversions::lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        // Find matching records
        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let mut updated = 0;

        for record in result.records() {
            let row = Self::record_to_row(record, &columns);
            let matches = predicate
                .map(|p| expressions::AmorphicExprEval::evaluate_predicate(&row, p))
                .unwrap_or(true);

            if matches {
                let updates: HashMap<String, AmorphicValue> = assignments
                    .iter()
                    .map(|(k, v)| (k.clone(), conversions::ast_to_amorphic(v)))
                    .collect();

                store
                    .update_fields(record.id, updates)
                    .map_err(conversions::amorphic_error)?;
                updated += 1;
            }
        }

        Ok(updated)
    }

    fn delete(&self, table: &str, predicate: Option<&Expression>) -> QueryResult<usize> {
        let mut store = self.store.write().map_err(conversions::lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        // Find matching records
        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let mut to_delete = Vec::new();

        for record in result.records() {
            let row = Self::record_to_row(record, &columns);
            let matches = predicate
                .map(|p| expressions::AmorphicExprEval::evaluate_predicate(&row, p))
                .unwrap_or(true);

            if matches {
                to_delete.push(record.id);
            }
        }

        let count = to_delete.len();
        for id in &to_delete {
            store.delete(*id).map_err(conversions::amorphic_error)?;
        }
        drop(store);

        // Clean up fulltext index entries for deleted records
        for id in &to_delete {
            let _ = self.update_fulltext_indexes_on_delete(table, *id);
        }

        Ok(count)
    }

    fn table_exists(&self, table: &str) -> QueryResult<bool> {
        let store = self.store.read().map_err(conversions::lock_error)?;
        let result =
            store.query_equals(SCHEMA_NAME_FIELD, &AmorphicValue::String(table.to_string()));
        Ok(result
            .records()
            .iter()
            .any(|r| r.get(TABLE_FIELD) == Some(&AmorphicValue::String(SCHEMA_TABLE.to_string()))))
    }

    fn index_scan(
        &self,
        table: &str,
        index: &str,
        predicate: &Expression,
    ) -> QueryResult<Vec<RowData>> {
        let store = self.store.read().map_err(conversions::lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        // Try to extract equality predicate for the indexed field
        if let Some((field, value)) = conversions::extract_eq_predicate(predicate) {
            if field == index {
                let amorphic_val = conversions::ast_to_amorphic(&value);
                let result = store.query_equals(&field, &amorphic_val);
                let rows = result
                    .records()
                    .iter()
                    .filter(|r| {
                        r.get(TABLE_FIELD) == Some(&AmorphicValue::String(table.to_string()))
                    })
                    .map(|r| Self::record_to_row(r, &columns))
                    .collect();

                return Ok(rows);
            }
        }

        // Try to extract range predicate for the indexed field
        if let Some((field, min, max)) = conversions::extract_range_predicate(predicate) {
            if field == index {
                let result = store.query_range(&field, min, max);
                let rows = result
                    .records()
                    .iter()
                    .filter(|r| {
                        r.get(TABLE_FIELD) == Some(&AmorphicValue::String(table.to_string()))
                    })
                    .map(|r| Self::record_to_row(r, &columns))
                    .collect();

                return Ok(rows);
            }
        }

        // Fallback: full scan with predicate filter
        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let rows = result
            .records()
            .iter()
            .map(|r| Self::record_to_row(r, &columns))
            .filter(|row| expressions::AmorphicExprEval::evaluate_predicate(row, predicate))
            .collect();

        Ok(rows)
    }
}
