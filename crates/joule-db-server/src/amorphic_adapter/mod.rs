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
pub(crate) use conversions::extract_conjunct_eq_predicate;
pub use conversions::{ast_to_amorphic_value, ast_to_json, expression_to_sql};
pub use crud::{AtomicBatchOp, DeferredBatchSync, PreparedAtomicBatch};
pub(crate) use expressions::evaluate_mvcc_predicate;

/// Reserved field name that identifies which SQL table a record belongs to
pub(super) const TABLE_FIELD: &str = "__table__";
/// The table tag field (columnar ingest batches carry it).
pub const TABLE_FIELD_NAME: &str = TABLE_FIELD;
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
    wal: std::sync::Arc<std::sync::Mutex<joule_db_local::storage::wal::WalManager>>,
    tx_counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
    next_id: std::sync::Arc<std::sync::atomic::AtomicU64>,
    pending: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// When bulk ingest last queued work (ms since `epoch`): the background
    /// catch-up waits for a short write-idle gap so it doesn't take the
    /// lock from a streaming COPY.
    last_bulk_ms: std::sync::Arc<std::sync::atomic::AtomicU64>,
    epoch: std::time::Instant,
    /// Group commit (fsync per commit, shared by concurrent writers) unless
    /// JOULEDB_SYNC_COMMIT=off. None = relaxed (flush to OS only).
    sync: Option<std::sync::Arc<joule_db_local::storage::wal::GroupSync>>,
    /// Writers blocked in `write()`: the background catch-up, which encodes
    /// under a shared lock, stops at its next slice so a bulk batch is not
    /// held behind a whole catch-up step.
    writers_waiting: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Columnar bulk batches committed but not yet row records (see
    /// `DurableAmorphicStore::materialize_pending`). Every lock handed out
    /// here except `write_bulk()`/`read_columns()` materializes them first,
    /// so row readers and writers always see every committed row; the
    /// background thread materializes them once writes go idle.
    cols_pending: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// Small pool (a quarter of the cores, at least one) for background
/// hologram catch-up, so it never takes every core from queries; its threads
/// run at a lower OS priority on Unix.
fn catchup_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let n = (std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            / 4)
        .max(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("jouledb-holo-{i}"))
            .start_handler(|_| {
                #[cfg(target_os = "linux")]
                unsafe {
                    // per-thread nice on Linux (tid as PRIO_PROCESS target)
                    libc::setpriority(libc::PRIO_PROCESS, libc::gettid() as libc::id_t, 10);
                }
                #[cfg(target_os = "macos")]
                unsafe {
                    libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0);
                }
            })
            .build()
            .expect("catch-up pool")
    })
}

/// `JOULEDB_SYNC_COMMIT`: `on` (default) makes every commit power-loss
/// durable before it is acknowledged; `off` acknowledges once the WAL is in
/// the OS page cache (survives a process crash, not a power loss).
pub fn sync_commit_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("JOULEDB_SYNC_COMMIT")
                .as_deref()
                .map(str::to_ascii_lowercase)
                .as_deref(),
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
                // sticky in GroupSync; the statement that owns this commit
                // turns it into an error via `take_commit_error`
                tracing::error!("commit durability: {e}");
                SYNC_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                COMMIT_ERROR.with(|c| c.replace(Some(format!("commit not durable: {e}"))));
            }
        }
    }
}

thread_local! {
    static COMMIT_ERROR: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Take (and clear) a durability failure raised by a commit on this thread.
/// Statement entry points call this before and after running, so a commit
/// whose fsync failed is reported to its client as an error.
pub fn take_commit_error() -> Option<String> {
    COMMIT_ERROR.with(|c| c.borrow_mut().take())
}

/// Tests: hold background materialization of columnar rows (readers and
/// writers still materialize what they need).
pub(crate) static PAUSE_BG_MATERIALIZE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

thread_local! {
    /// Store read guards this thread holds (see `HoloLock::settle_columns`).
    static READS_HELD: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Read guard of the store (counts held guards per thread).
pub(crate) struct StoreReadGuard<'a> {
    guard: std::sync::RwLockReadGuard<'a, DurableAmorphicStore>,
}

impl std::ops::Deref for StoreReadGuard<'_> {
    type Target = DurableAmorphicStore;
    fn deref(&self) -> &DurableAmorphicStore {
        &self.guard
    }
}

impl Drop for StoreReadGuard<'_> {
    fn drop(&mut self) {
        READS_HELD.with(|c| c.set(c.get().saturating_sub(1)));
    }
}

fn wrap_read(
    r: std::sync::LockResult<std::sync::RwLockReadGuard<'_, DurableAmorphicStore>>,
) -> std::sync::LockResult<StoreReadGuard<'_>> {
    let wrap = |guard| {
        READS_HELD.with(|c| c.set(c.get() + 1));
        StoreReadGuard { guard }
    };
    match r {
        Ok(g) => Ok(wrap(g)),
        Err(p) => Err(std::sync::PoisonError::new(wrap(p.into_inner()))),
    }
}

/// fsync failures seen by commit guards (metric; also sticky in GroupSync)
pub static SYNC_FAILURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl HoloLock {
    /// Records per background catch-up step (bounds how long it holds the lock).
    const STEP: usize = 2_048;
    /// Columnar rows materialized per background step.
    const MATERIALIZE_STEP: usize = 16_384;
    /// Write-idle gap before the background catch-up runs.
    const IDLE_MS: u64 = 250;

    fn new(store: DurableAmorphicStore) -> Self {
        let wal = store.wal_handle();
        let tx_counter = store.tx_counter_handle();
        let next_id = store.next_id_handle();
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
        let cols_pending = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cols_flag = cols_pending.clone();
        let writers_waiting = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let waiting = writers_waiting.clone();
        let epoch = std::time::Instant::now();
        let _ = std::thread::Builder::new()
            .name("jouledb-holo-catchup".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    let Some(lock) = weak.upgrade() else { return };
                    // columnar rows -> row records first (bounded steps,
                    // only when bulk writes are idle)
                    if cols_flag.load(std::sync::atomic::Ordering::Acquire)
                        && !PAUSE_BG_MATERIALIZE.load(std::sync::atomic::Ordering::Relaxed)
                    {
                        let now = epoch.elapsed().as_millis() as u64;
                        if now.saturating_sub(last.load(std::sync::atomic::Ordering::Relaxed))
                            >= Self::IDLE_MS
                        {
                            if let Ok(mut g) = lock.try_write() {
                                let _ = g.materialize_pending(Self::MATERIALIZE_STEP);
                                if g.pending_rows() == 0 {
                                    cols_flag.store(false, std::sync::atomic::Ordering::Release);
                                }
                                if g.pending_holograms() > 0 {
                                    flag.store(true, std::sync::atomic::Ordering::Release);
                                }
                            }
                        }
                        continue;
                    }
                    if !flag.load(std::sync::atomic::Ordering::Acquire) {
                        continue;
                    }
                    let now = epoch.elapsed().as_millis() as u64;
                    if now.saturating_sub(last.load(std::sync::atomic::Ordering::Relaxed))
                        < Self::IDLE_MS
                    {
                        continue;
                    }
                    // Encode the next step under a *read* lock in a small
                    // low-priority pool (queries keep running and keep their
                    // cores), then take the write lock only to apply it.
                    let plan = match lock.read() {
                        Ok(g) => g.plan_catch_up_until(
                            Self::STEP,
                            &|| waiting.load(std::sync::atomic::Ordering::Relaxed) > 0,
                            |f| catchup_pool().install(f),
                        ),
                        Err(_) => None,
                    };
                    if let Some((ids, holos)) = plan {
                        if ids.is_empty() {
                            flag.store(false, std::sync::atomic::Ordering::Release);
                            continue;
                        }
                        let mut plan = Some((ids, holos));
                        for _ in 0..20 {
                            if let Ok(mut g) = lock.try_write() {
                                let now = epoch.elapsed().as_millis() as u64;
                                if now
                                    .saturating_sub(last.load(std::sync::atomic::Ordering::Relaxed))
                                    >= Self::IDLE_MS
                                {
                                    if let Some((ids, holos)) = plan.take() {
                                        g.apply_catch_up_plan(ids, holos);
                                        if g.pending_holograms() == 0 {
                                            flag.store(false, std::sync::atomic::Ordering::Release);
                                        }
                                    }
                                }
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_micros(500));
                        }
                        continue;
                    }
                    // a field vector must be created first: one exclusive step.
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
            wal,
            tx_counter,
            next_id,
            pending,
            last_bulk_ms,
            epoch,
            sync,
            writers_waiting,
            cols_pending,
        }
    }

    /// Construct a lightweight pure-RAM `HoloLock` without `GroupSync` or background OS thread
    /// spawning — used by ephemeral Copy-on-Write SQL branch sandboxes (`CREATE BRANCH` dry-runs).
    fn new_ephemeral(store: DurableAmorphicStore) -> Self {
        let wal = store.wal_handle();
        let tx_counter = store.tx_counter_handle();
        let next_id = store.next_id_handle();
        Self {
            inner: std::sync::Arc::new(RwLock::new(store)),
            wal,
            tx_counter,
            next_id,
            pending: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            last_bulk_ms: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            epoch: std::time::Instant::now(),
            sync: None,
            writers_waiting: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            cols_pending: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    pub(crate) fn settle_if_cols_pending(&self) {
        self.settle_columns();
    }

    pub(crate) fn alloc_tx_and_ids(&self, n_inserts: usize) -> (u64, u64) {
        let tx_id = self
            .tx_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let first_id = if n_inserts > 0 {
            self.next_id
                .fetch_add(n_inserts as u64, std::sync::atomic::Ordering::SeqCst)
        } else {
            0
        };
        (tx_id, first_id)
    }

    /// Write and flush an atomic OLTP transaction to `WalManager` WITHOUT acquiring
    /// the exclusive `RwLock` on `DurableAmorphicStore` (zero SQL reader starvation).
    pub(crate) fn log_wal_tx_unlocked(
        &self,
        tx_id: u64,
        pages: &[(u64, Vec<u8>)],
    ) -> Result<
        (
            u64,
            Option<std::sync::Arc<joule_db_local::storage::wal::GroupSync>>,
        ),
        String,
    > {
        let sync = self.sync.clone();
        if sync.as_ref().is_some_and(|s| s.has_failed()) {
            return Err("WAL fsync failed earlier".into());
        }
        let mut wal_guard = self.wal.lock().unwrap_or_else(|p| p.into_inner());
        let commit_lsn = wal_guard
            .log_atomic_tx_fast(tx_id, pages)
            .map_err(|e| format!("WAL commit failed: {e}"))?;
        drop(wal_guard);
        Ok((commit_lsn, sync))
    }

    pub(crate) fn write_ram_only(
        &self,
    ) -> std::sync::LockResult<std::sync::RwLockWriteGuard<'_, DurableAmorphicStore>> {
        self.writers_waiting
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let w = self.inner.write();
        self.writers_waiting
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        match w {
            Ok(mut g) => {
                if self.cols_pending.load(std::sync::atomic::Ordering::Acquire) {
                    self.settle_locked(&mut g);
                }
                Ok(g)
            }
            Err(p) => Err(p),
        }
    }

    /// Materialize queued columnar rows (exclusive lock) if there are any.
    /// A thread that already holds a read guard of this store cannot wait
    /// for the write lock (it would wait on itself); its outer acquisition
    /// already settled, so it only settles if the lock is free.
    fn settle_columns(&self) {
        if self.cols_pending.load(std::sync::atomic::Ordering::Acquire) {
            if READS_HELD.with(|c| c.get()) > 0 {
                if let Ok(mut g) = self.inner.try_write() {
                    self.settle_locked(&mut g);
                }
            } else if let Ok(mut g) = self.inner.write() {
                self.settle_locked(&mut g);
            }
        }
    }

    fn settle_locked(&self, g: &mut DurableAmorphicStore) {
        if g.pending_rows() > 0 {
            if let Err(e) = g.settle_pending() {
                tracing::error!("materializing bulk rows: {e}");
                return;
            }
            if g.pending_holograms() > 0 {
                self.pending
                    .store(true, std::sync::atomic::Ordering::Release);
            }
        }
        self.cols_pending
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Write lock for columnar bulk ingest: queued columnar rows stay queued
    /// (the new batch queues behind them).
    pub(crate) fn write_bulk(&self) -> std::sync::LockResult<SyncWriteGuard<'_>> {
        self.write_inner(false)
    }

    /// Note that columnar ingest queued rows (call with the bulk lock held).
    pub(crate) fn mark_columns_pending(&self) {
        self.last_bulk_ms.store(
            self.epoch.elapsed().as_millis() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.cols_pending
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Read lock that does not materialize queued columnar rows: only for
    /// readers that never look at row records of user tables (column
    /// snapshots keyed by `generation()`, which counts queued rows; schema
    /// metadata, which is always written through the row path).
    pub(crate) fn read_columns(&self) -> std::sync::LockResult<StoreReadGuard<'_>> {
        wrap_read(self.inner.read())
    }

    pub(crate) fn write(&self) -> std::sync::LockResult<SyncWriteGuard<'_>> {
        self.write_inner(true)
    }

    /// Acquire the exclusive write lock while returning a cloned `GroupSync` handle
    /// instead of waiting on `Drop` inside the caller's critical section.
    pub(crate) fn write_deferred_sync(
        &self,
    ) -> std::sync::LockResult<(
        std::sync::RwLockWriteGuard<'_, DurableAmorphicStore>,
        Option<std::sync::Arc<joule_db_local::storage::wal::GroupSync>>,
    )> {
        let sync = self.sync.clone();
        self.writers_waiting
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let w = self.inner.write();
        self.writers_waiting
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        match w {
            Ok(g) if sync.as_ref().is_some_and(|s| s.has_failed()) => {
                Err(std::sync::PoisonError::new((g, None)))
            }
            Ok(mut g) => {
                if self.cols_pending.load(std::sync::atomic::Ordering::Acquire) {
                    self.settle_locked(&mut g);
                }
                Ok((g, sync))
            }
            Err(p) => Err(std::sync::PoisonError::new((p.into_inner(), sync))),
        }
    }

    fn write_inner(&self, settle: bool) -> std::sync::LockResult<SyncWriteGuard<'_>> {
        let sync = self.sync.as_deref();
        self.writers_waiting
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let w = self.inner.write();
        self.writers_waiting
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        match w {
            // after an fsync failure, refuse further writes (nothing more is
            // acknowledged on a WAL that may not be durable)
            Ok(g) if sync.is_some_and(|s| s.has_failed()) => {
                Err(std::sync::PoisonError::new(SyncWriteGuard {
                    guard: Some(g),
                    sync: None,
                }))
            }
            Ok(mut g) => {
                if settle && self.cols_pending.load(std::sync::atomic::Ordering::Acquire) {
                    self.settle_locked(&mut g);
                }
                Ok(SyncWriteGuard {
                    guard: Some(g),
                    sync,
                })
            }
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
    pub(crate) fn read(&self) -> std::sync::LockResult<StoreReadGuard<'_>> {
        self.settle_columns();
        wrap_read(self.inner.read())
    }

    /// Read lock for hypervector consumers (similarity search): finishes any
    /// queued hypervector work first so results cover every committed row.
    pub(crate) fn read_indexed(&self) -> std::sync::LockResult<StoreReadGuard<'_>> {
        self.settle_columns();
        if self.pending.load(std::sync::atomic::Ordering::Acquire) {
            if let Ok(mut g) = self.inner.write() {
                g.catch_up_holograms(usize::MAX);
                self.pending
                    .store(false, std::sync::atomic::Ordering::Release);
            }
        }
        wrap_read(self.inner.read())
    }

    pub(crate) fn read_primary(&self) -> std::sync::LockResult<StoreReadGuard<'_>> {
        self.settle_columns();
        wrap_read(self.inner.read())
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

#[derive(Debug, Clone)]
pub(crate) struct CachedTableSchema {
    pub columns: std::sync::Arc<Vec<String>>,
    pub defs: std::sync::Arc<Vec<ColumnDefInfo>>,
    pub has_computed: bool,
}

pub struct AmorphicTableStorage {
    store: HoloLock,
    db_path: std::path::PathBuf,
    /// Cached `(columns, column_defs, has_computed)` per table, invalidated only on DDL/schema mutation.
    pub(crate) schema_cache: std::sync::RwLock<HashMap<String, CachedTableSchema>>,
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
    /// Out-of-core `.jcol` segmented tables for large analytical datasets (`> 262K` rows).
    disk_cols: std::sync::RwLock<
        HashMap<String, std::sync::Arc<crate::disk_col_segments::DiskColumnTable>>,
    >,
}

impl AmorphicTableStorage {
    /// Group-commit handle (None when JOULEDB_SYNC_COMMIT=off).
    pub fn group_sync(&self) -> Option<&joule_db_local::storage::wal::GroupSync> {
        self.store.group_sync()
    }

    pub fn new(store: DurableAmorphicStore) -> Self {
        let db_path = store.path().to_path_buf();
        let recovered_disk = crate::disk_col_segments::DiskColumnTable::open_all(&db_path);
        Self {
            store: HoloLock::new(store),
            db_path,
            schema_cache: std::sync::RwLock::new(HashMap::new()),
            col_cache: std::sync::Mutex::new(HashMap::new()),
            disk_cols: std::sync::RwLock::new(recovered_disk),
        }
    }

    /// Construct a zero-FD, zero-thread pure-RAM `AmorphicTableStorage` for ephemeral CoW SQL branches.
    pub fn new_ephemeral_branch() -> Self {
        let store = DurableAmorphicStore::new_in_memory();
        Self {
            store: HoloLock::new_ephemeral(store),
            db_path: std::path::PathBuf::new(),
            schema_cache: std::sync::RwLock::new(HashMap::new()),
            col_cache: std::sync::Mutex::new(HashMap::new()),
            disk_cols: std::sync::RwLock::new(HashMap::new()),
        }
    }

    #[inline]
    pub(crate) fn invalidate_schema_cache(&self, table: &str) {
        if let Ok(mut sc) = self.schema_cache.write() {
            sc.remove(table);
        }
    }

    pub(crate) fn cached_schema_with_store(
        &self,
        store: &DurableAmorphicStore,
        table: &str,
    ) -> QueryResult<CachedTableSchema> {
        if let Ok(sc) = self.schema_cache.read()
            && let Some(hit) = sc.get(table)
        {
            return Ok(hit.clone());
        }
        let cols = self.load_schema_columns_uncached(store, table)?;
        let defs = self
            .load_column_defs_uncached(store, table)
            .unwrap_or_default();
        let has_computed = defs.iter().any(|d| d.computed_expr.is_some());
        let entry = CachedTableSchema {
            columns: std::sync::Arc::new(cols),
            defs: std::sync::Arc::new(defs),
            has_computed,
        };
        if let Ok(mut sc) = self.schema_cache.write() {
            if sc.len() > 512 {
                sc.clear();
            }
            sc.insert(table.to_string(), entry.clone());
        }
        Ok(entry)
    }

    #[inline]
    pub fn has_computed_columns(&self, table: &str) -> bool {
        if let Ok(sc) = self.schema_cache.read()
            && let Some(hit) = sc.get(table)
        {
            return hit.has_computed;
        }
        let Ok(store) = self.store.read_columns() else {
            return false;
        };
        self.cached_schema_with_store(&store, table)
            .is_ok_and(|c| c.has_computed)
    }

    /// Return the out-of-core `.jcol` segmented handle for `table` if spilled to disk.
    pub fn disk_column_table(
        &self,
        table: &str,
    ) -> Option<std::sync::Arc<crate::disk_col_segments::DiskColumnTable>> {
        self.disk_cols.read().ok()?.get(table).cloned()
    }

    /// Columnar rows not yet row records (tests and diagnostics).
    pub fn pending_column_rows(&self) -> usize {
        self.store
            .read_columns()
            .map(|s| s.pending_rows())
            .unwrap_or(0)
    }

    /// Materialize up to `max` queued columnar rows now (tests).
    #[cfg(test)]
    pub(crate) fn materialize_columns(&self, max: usize) -> usize {
        let mut g = self.store.write_bulk().unwrap();
        g.materialize_pending(max).unwrap()
    }

    /// Return `Some(cols)` ONLY if `col_cache` already holds a snapshot matching
    /// the current store generation (never rebuilds or settles columns).
    pub fn cached_table_columns(
        &self,
        table: &str,
    ) -> Option<std::sync::Arc<crate::columnar_exec::TableColumns>> {
        if self.get_view(table).is_some() {
            return None;
        }
        let store = self.store.read_columns().ok()?;
        let generation = store.generation();
        if let Some((g, t)) = self.col_cache.lock().ok()?.get(table) {
            if *g == generation {
                return Some(t.clone());
            }
        }
        None
    }

    /// When `col_cache` missed (e.g. during OLTP writes), check whether `where_clause`
    /// contains a type-exact `col = literal` equality conjunct matching at most `max_rows`
    /// records via `AmorphicStore::records_equal`, and if so build a small ephemeral
    /// `TableColumns` over just those `<= max_rows` borrowed records in O(k) instead of
    /// rebuilding the entire table.
    pub fn try_selective_table_columns(
        &self,
        table: &str,
        where_clause: Option<&Expression>,
        max_rows: usize,
    ) -> Option<std::sync::Arc<crate::columnar_exec::TableColumns>> {
        if self.get_view(table).is_some() {
            return None;
        }
        let pred = where_clause?;
        let (col, val) = conversions::extract_conjunct_eq_predicate(pred)?;
        if !matches!(
            val,
            AstValue::Int(_) | AstValue::String(_) | AstValue::Bool(_)
        ) {
            return None;
        }
        let store = self.store.read_primary().ok()?;
        let columns = self.get_schema_columns(&store, table).ok()?;
        if !columns.iter().any(|c| c == &col) {
            return None;
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
                    return None;
                }
            } else if matches!(val, AstValue::Int(_)) {
                return None;
            }
        }
        let amorphic_val = conversions::ast_to_amorphic(&val);
        let table_tag = AmorphicValue::String(table.to_string());
        let recs: Vec<&AmorphicRecord> = store
            .records_equal(&col, &amorphic_val)
            .into_iter()
            .filter(|r| r.get(TABLE_FIELD) == Some(&table_tag))
            .collect();
        if recs.len() > max_rows {
            return None;
        }
        Some(std::sync::Arc::new(
            crate::columnar_exec::TableColumns::build(&columns, &recs),
        ))
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
        // A current snapshot covers queued columnar rows too (generation
        // counts them), so a hit does not materialize them.
        {
            let store = self.store.read_columns().ok()?;
            let generation = store.generation();
            if let Some((g, t)) = self.col_cache.lock().ok()?.get(table) {
                if *g == generation {
                    return Some(t.clone());
                }
            }
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
        let store = match self.store.read_columns() {
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

    /// Get schema columns for a table (O(1) cached; falls back to `__schemas__` lookup on first touch).
    pub(super) fn get_schema_columns(
        &self,
        store: &DurableAmorphicStore,
        table: &str,
    ) -> QueryResult<Vec<String>> {
        if let Ok(sc) = self.schema_cache.read()
            && let Some(hit) = sc.get(table)
        {
            return Ok(hit.columns.as_ref().clone());
        }
        Ok(self
            .cached_schema_with_store(store, table)?
            .columns
            .as_ref()
            .clone())
    }

    pub(super) fn load_schema_columns_uncached(
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

    /// Direct O(1)/O(k) indexed equality `SELECT` fast-path for OLTP point reads and
    /// indexed aggregate probes (`SELECT c1, c2 FROM t WHERE k = lit` or
    /// `SELECT COUNT(*), SUM(c) FROM t WHERE k = lit`) when `col_cache` is invalidated
    /// by concurrent writes — avoiding `TableColumns::build` allocations inside `store.read()`.
    pub fn try_direct_indexed_eq_query(
        &self,
        query: &joule_db_query::sql::SqlQuery,
        request_limit: Option<usize>,
    ) -> Option<(Vec<String>, Vec<Vec<serde_json::Value>>, bool)> {
        if !query.joins.is_empty()
            || !query.ctes.is_empty()
            || !query.group_by.is_empty()
            || query.having.is_some()
            || !query.order_by.is_empty()
            || query.distinct
            || query.set_op.is_some()
            || query.as_of_timestamp.is_some()
        {
            return None;
        }
        let table = query.from.as_ref()?.table_name()?;
        if self.get_view(table).is_some() || self.disk_column_table(table).is_some() {
            return None;
        }
        let where_expr = query.where_clause.as_ref()?;
        // Must be a single exact `col = lit` predicate (not a multi-conjunct filter)
        let (eq_col, mut eq_val) = conversions::extract_eq_predicate(where_expr)?;
        if !matches!(
            eq_val,
            AstValue::Int(_) | AstValue::String(_) | AstValue::Bool(_)
        ) {
            return None;
        }

        // Resolve cached schema without holding `store.read()` if already cached
        let schema = if let Ok(sc) = self.schema_cache.read()
            && let Some(hit) = sc.get(table)
        {
            hit.clone()
        } else {
            let store = self.store.read_columns().ok()?;
            self.cached_schema_with_store(&store, table).ok()?
        };
        if schema.has_computed || !schema.columns.iter().any(|c| c == &eq_col) {
            return None;
        }
        if let Some(def) = schema.defs.iter().find(|d| d.name == eq_col) {
            let dt = def.data_type.to_uppercase();
            let is_int_col = matches!(
                dt.as_str(),
                "INT" | "INTEGER" | "BIGINT" | "SMALLINT" | "SERIAL" | "BIGSERIAL"
            );
            if is_int_col
                && let AstValue::String(ref s) = eq_val
                && let Ok(parsed_i) = s.trim().parse::<i64>()
            {
                eq_val = AstValue::Int(parsed_i);
            }
            let type_ok = match eq_val {
                AstValue::Int(_) => is_int_col,
                AstValue::String(_) => {
                    matches!(dt.as_str(), "TEXT" | "VARCHAR" | "CHAR" | "STRING" | "UUID")
                }
                AstValue::Bool(_) => matches!(dt.as_str(), "BOOL" | "BOOLEAN"),
                _ => false,
            };
            if !type_ok && !dt.is_empty() {
                return None;
            }
        } else if matches!(eq_val, AstValue::Int(_)) {
            return None;
        }

        // Classify SELECT list: either pure simple column projections (`c1, c2` / `*`)
        // OR pure simple aggregates (`COUNT(*)`, `COUNT(c)`, `SUM(c)`, `AVG(c)`, `MIN(c)`, `MAX(c)`)
        enum DirectAggKind {
            CountStar,
            Count(String),
            Sum(String),
            Avg(String),
            Min(String),
            Max(String),
        }

        let mut proj_cols: Vec<(String, String)> = Vec::new(); // (src_col, out_name)
        let mut agg_specs: Vec<(DirectAggKind, String)> = Vec::new();
        let mut saw_col = false;
        let mut saw_agg = false;

        for sc in &query.columns {
            match &sc.expr {
                Expression::Wildcard => {
                    saw_col = true;
                    for c in schema.columns.iter() {
                        proj_cols.push((c.clone(), c.clone()));
                    }
                }
                Expression::Column(c) => {
                    let real = schema.columns.iter().find(|x| x.eq_ignore_ascii_case(c))?;
                    saw_col = true;
                    proj_cols.push((
                        real.clone(),
                        sc.alias.clone().unwrap_or_else(|| real.clone()),
                    ));
                }
                Expression::QualifiedColumn { column, .. } => {
                    let real = schema
                        .columns
                        .iter()
                        .find(|x| x.eq_ignore_ascii_case(column))?;
                    saw_col = true;
                    proj_cols.push((
                        real.clone(),
                        sc.alias.clone().unwrap_or_else(|| real.clone()),
                    ));
                }
                Expression::Function { name, args } => {
                    saw_agg = true;
                    let up = name.to_ascii_uppercase();
                    if up == "COUNT" && args.len() == 1 && matches!(args[0], Expression::Wildcard) {
                        let out_name = sc.alias.clone().unwrap_or_else(|| format!("{name}(*)"));
                        agg_specs.push((DirectAggKind::CountStar, out_name));
                    } else if args.len() == 1 {
                        let c_raw = match &args[0] {
                            Expression::Column(c) => c.as_str(),
                            Expression::QualifiedColumn { column, .. } => column.as_str(),
                            _ => return None,
                        };
                        let real = schema
                            .columns
                            .iter()
                            .find(|x| x.eq_ignore_ascii_case(c_raw))?
                            .clone();
                        let out_name = sc
                            .alias
                            .clone()
                            .unwrap_or_else(|| format!("{name}({real})"));
                        let kind = match up.as_str() {
                            "COUNT" => DirectAggKind::Count(real),
                            "SUM" => DirectAggKind::Sum(real),
                            "AVG" => DirectAggKind::Avg(real),
                            "MIN" => DirectAggKind::Min(real),
                            "MAX" => DirectAggKind::Max(real),
                            _ => return None,
                        };
                        agg_specs.push((kind, out_name));
                    } else {
                        return None;
                    }
                }
                _ => return None,
            }
        }
        if saw_col == saw_agg {
            return None;
        }

        let amorphic_val = conversions::ast_to_amorphic(&eq_val);
        let table_tag = AmorphicValue::String(table.to_string());

        if saw_col {
            let off = query.offset.unwrap_or(0);
            let sql_lim = query.limit;
            let store = self.store.read().ok()?;
            let mut rows = Vec::with_capacity(sql_lim.unwrap_or(8).min(64));
            let mut skipped = 0usize;
            let mut matched_total = 0usize;
            let mut truncated = false;
            let mut exceeded_unbounded = false;
            store.for_each_record_equal(&eq_col, &amorphic_val, |r| {
                if r.get(TABLE_FIELD) != Some(&table_tag) {
                    return true;
                }
                matched_total += 1;
                if sql_lim.is_none() && matched_total > 256 {
                    exceeded_unbounded = true;
                    return false;
                }
                if skipped < off {
                    skipped += 1;
                    return true;
                }
                if let Some(lim) = sql_lim
                    && rows.len() >= lim
                {
                    return false;
                }
                if sql_lim.is_none()
                    && let Some(rlim) = request_limit
                    && rows.len() >= rlim
                {
                    truncated = true;
                    return false;
                }
                let mut row = Vec::with_capacity(proj_cols.len());
                for (src_c, _) in &proj_cols {
                    let jv = r
                        .get(src_c)
                        .map(conversions::amorphic_value_to_json)
                        .unwrap_or(serde_json::Value::Null);
                    row.push(jv);
                }
                rows.push(row);
                true
            });
            if exceeded_unbounded {
                return None;
            }
            let out_names: Vec<String> = proj_cols.iter().map(|(_, o)| o.clone()).collect();
            Some((out_names, rows, truncated))
        } else {
            #[derive(Default)]
            struct RegAcc {
                cnt: u64,
                isum: i128,
                fsum: f64,
                saw_float: bool,
                imin: i64,
                imax: i64,
                fmin: f64,
                fmax: f64,
            }
            let mut accs: Vec<RegAcc> = (0..agg_specs.len())
                .map(|_| RegAcc {
                    cnt: 0,
                    isum: 0,
                    fsum: 0.0,
                    saw_float: false,
                    imin: i64::MAX,
                    imax: i64::MIN,
                    fmin: f64::INFINITY,
                    fmax: f64::NEG_INFINITY,
                })
                .collect();

            let store = self.store.read().ok()?;
            let mut non_numeric_abort = false;
            store.for_each_record_equal(&eq_col, &amorphic_val, |r| {
                if r.get(TABLE_FIELD) != Some(&table_tag) {
                    return true;
                }
                for (ai, (kind, _)) in agg_specs.iter().enumerate() {
                    let acc = &mut accs[ai];
                    match kind {
                        DirectAggKind::CountStar => acc.cnt += 1,
                        DirectAggKind::Count(c) => {
                            if !matches!(r.get(c), None | Some(AmorphicValue::Null)) {
                                acc.cnt += 1;
                            }
                        }
                        DirectAggKind::Sum(c)
                        | DirectAggKind::Avg(c)
                        | DirectAggKind::Min(c)
                        | DirectAggKind::Max(c) => match r.get(c) {
                            None | Some(AmorphicValue::Null) => {}
                            Some(AmorphicValue::Int(v)) => {
                                acc.cnt += 1;
                                acc.isum += *v as i128;
                                acc.imin = acc.imin.min(*v);
                                acc.imax = acc.imax.max(*v);
                                let fv = *v as f64;
                                acc.fmin = acc.fmin.min(fv);
                                acc.fmax = acc.fmax.max(fv);
                            }
                            Some(AmorphicValue::Float(fv)) => {
                                acc.cnt += 1;
                                acc.saw_float = true;
                                acc.fsum += *fv;
                                acc.fmin = acc.fmin.min(*fv);
                                acc.fmax = acc.fmax.max(*fv);
                            }
                            _ => {
                                non_numeric_abort = true;
                                return false;
                            }
                        },
                    }
                }
                true
            });
            drop(store);
            if non_numeric_abort {
                return None;
            }

            let out_names: Vec<String> = agg_specs.iter().map(|(_, o)| o.clone()).collect();
            let mut out_row = Vec::with_capacity(agg_specs.len());
            for ((kind, _), acc) in agg_specs.iter().zip(accs.iter()) {
                let v = match kind {
                    DirectAggKind::CountStar | DirectAggKind::Count(_) => {
                        serde_json::Value::Number(serde_json::Number::from(acc.cnt))
                    }
                    DirectAggKind::Sum(_) => {
                        if acc.cnt == 0 {
                            serde_json::Value::Null
                        } else if acc.saw_float {
                            let tot = acc.fsum + (acc.isum as f64);
                            serde_json::Number::from_f64(tot)
                                .map(serde_json::Value::Number)
                                .unwrap_or(serde_json::Value::Null)
                        } else if let Ok(i) = i64::try_from(acc.isum) {
                            serde_json::Value::Number(serde_json::Number::from(i))
                        } else {
                            serde_json::Number::from_f64(acc.isum as f64)
                                .map(serde_json::Value::Number)
                                .unwrap_or(serde_json::Value::Null)
                        }
                    }
                    DirectAggKind::Avg(_) => {
                        if acc.cnt == 0 {
                            serde_json::Value::Null
                        } else {
                            let tot = acc.fsum + (acc.isum as f64);
                            serde_json::Number::from_f64(tot / (acc.cnt as f64))
                                .map(serde_json::Value::Number)
                                .unwrap_or(serde_json::Value::Null)
                        }
                    }
                    DirectAggKind::Min(_) => {
                        if acc.cnt == 0 {
                            serde_json::Value::Null
                        } else if acc.saw_float {
                            serde_json::Number::from_f64(acc.fmin)
                                .map(serde_json::Value::Number)
                                .unwrap_or(serde_json::Value::Null)
                        } else {
                            serde_json::Value::Number(serde_json::Number::from(acc.imin))
                        }
                    }
                    DirectAggKind::Max(_) => {
                        if acc.cnt == 0 {
                            serde_json::Value::Null
                        } else if acc.saw_float {
                            serde_json::Number::from_f64(acc.fmax)
                                .map(serde_json::Value::Number)
                                .unwrap_or(serde_json::Value::Null)
                        } else {
                            serde_json::Value::Number(serde_json::Number::from(acc.imax))
                        }
                    }
                };
                out_row.push(v);
            }
            let off = query.offset.unwrap_or(0);
            let rows = if off > 0 || query.limit == Some(0) {
                Vec::new()
            } else {
                vec![out_row]
            };
            Some((out_names, rows, false))
        }
    }
}

// ============ TableStorage implementation ============

impl TableStorage for AmorphicTableStorage {
    fn scan(&self, table: &str) -> QueryResult<Vec<RowData>> {
        let store = self.store.read().map_err(conversions::lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        let mut rows: Vec<RowData> = store
            .records_equal(TABLE_FIELD, &AmorphicValue::String(table.to_string()))
            .into_iter()
            .map(|r| Self::record_to_row(r, &columns))
            .collect();
        drop(store);

        if let Some(dt) = self.disk_column_table(table) {
            let disk_rows = dt.scan_rows(&columns).map_err(|e| {
                QueryError::ExecutionError(format!("disk segment scan failed: {e}"))
            })?;
            if rows.is_empty() {
                return Ok(disk_rows);
            }
            rows.extend(disk_rows);
        }

        Ok(rows)
    }

    fn columns(&self, table: &str) -> QueryResult<Vec<String>> {
        if let Ok(sc) = self.schema_cache.read()
            && let Some(hit) = sc.get(table)
        {
            return Ok(hit.columns.as_ref().clone());
        }
        let store = self.store.read_columns().map_err(conversions::lock_error)?;
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
        let store = self.store.read_columns().map_err(conversions::lock_error)?;
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
