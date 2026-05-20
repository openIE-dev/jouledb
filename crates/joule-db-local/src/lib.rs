//! # JouleDB Local
//!
//! Native/local storage backend for JouleDB with file support.
//!
//! ## Features
//!
//! - `file` (default) - Simple file-based storage with WAL
//! - `mmap` - Memory-mapped file support
//! - `server` - TCP server with binary protocol
//!
//! ## Usage
//!
//! ```rust,ignore
//! use joule_db_local::{Database, FileBackend};
//!
//! let db = Database::open("./mydb")?;
//! db.put(b"key", b"value")?;
//! let value = db.get(b"key")?;
//! db.sync()?;
//! ```
//!
//! ## Transaction Usage
//!
//! ```rust,ignore
//! use joule_db_local::Database;
//!
//! let db = Database::open("./mydb")?;
//!
//! // Explicit transaction
//! let tx = db.begin()?;
//! tx.put(b"key1", b"value1")?;
//! tx.put(b"key2", b"value2")?;
//! tx.commit()?;
//! ```
//!
//! ## Server Usage
//!
//! ```rust,ignore
//! use joule_db_local::{Database, server::TcpServer};
//!
//! #[tokio::main]
//! async fn main() {
//!     let db = Database::open("./mydb").unwrap();
//!     let server = TcpServer::new(db);
//!     server.run("127.0.0.1:6380").await.unwrap();
//! }
//! ```

pub mod bloom;
pub mod lsm;
pub mod recovery;
pub mod snapshot_registry;
pub mod storage;

#[cfg(feature = "server")]
pub mod server;

pub use recovery::{
    CheckpointConfig, CheckpointRecord, DirtyPageEntry, PageLsnTracker,
    RecoveryManager as AriesRecoveryManager, RecoveryStats, TransactionEntry, TransactionState,
};
pub use storage::file::FileBackend;
pub use storage::wal::{
    RecoveryManager as WalRecoveryManager, RecoveryResult, WalEntry, WalEntryType, WalManager,
};

#[cfg(feature = "server")]
pub use server::{
    ServerStats, ServerStatsSnapshot, TcpClient, TcpClientConfig, TcpServer, TcpServerConfig,
};

pub use bloom::BloomFilter;
pub use joule_db_core::Error;

use joule_db_core::index::Index;
use joule_db_core::{Engine, WriteTransaction};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Prefix for TTL metadata keys in the B-tree.
const TTL_PREFIX: &[u8] = b"__ttl__::";

/// Get the current Unix timestamp in seconds.
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Convert a slice into a fixed-size byte array, returning a typed
/// `Error::Storage` instead of panicking when the slice is the wrong
/// length. Used for TTL parsing where the on-disk format is 8 bytes
/// big-endian but a corrupt or truncated row could give us less.
fn slice_to_array<const N: usize>(s: &[u8]) -> Result<[u8; N], Error> {
    s.try_into().map_err(|_| {
        Error::Storage(joule_db_core::error::StorageError::Backend(format!(
            "expected {} bytes, got {}",
            N,
            s.len()
        )))
    })
}

/// Build the TTL metadata key for a given user key.
fn ttl_key(key: &[u8]) -> Vec<u8> {
    let mut k = Vec::with_capacity(TTL_PREFIX.len() + key.len());
    k.extend_from_slice(TTL_PREFIX);
    k.extend_from_slice(key);
    k
}

/// Local database instance with WAL-based crash recovery
///
/// The Database provides a high-level API for key-value operations with:
/// - Automatic crash recovery via Write-Ahead Logging
/// - Transaction support with commit/rollback
/// - Checkpoint management
/// - Thread-safe operations
pub struct Database {
    /// The B-tree engine (owns the FileBackend)
    engine: Arc<Engine>,
    /// Database path
    path: String,
    /// **Writer-exclusion lockfile (Phase 8 of CoW MVCC).** When
    /// the database was opened via `open_writer`, this holds an
    /// exclusive `flock` on `<db>/writer.lock`. Drop releases the
    /// lock so a peer process can subsequently open as the writer.
    ///
    /// `None` for read-only opens (`Database::open`), or for
    /// in-memory test databases where cross-process exclusion has
    /// no meaning.
    ///
    /// **Why this exists**: prior to 2026-04-30, two writer
    /// processes could both successfully `Database::open` the
    /// same path and proceed to write concurrently. Each had its
    /// own `FileBackend` with its own `next_page_id` allocator;
    /// they'd hand out the same fresh page ids and trample each
    /// other at the file level, corrupting the catalog. ROR ingest
    /// + the supervisor's hourly daemon tick reproduced this for
    /// 9 hours straight. See `docs/scholar/CORRUPTION-INVESTIGATION-2026-04-28.md`
    /// for the postmortem.
    _writer_lock: Option<std::fs::File>,
}

impl Database {
    /// Open or create a database at the given path **without
    /// taking the writer-exclusion lock**.
    ///
    /// Use this for **read-only** workloads — `engine.get`,
    /// `engine.range`, `engine.scan`. Calling `engine.put` or any
    /// other write through a Database opened with `open` is a
    /// **logic bug**: a peer process holding the writer lock will
    /// silently get its writes corrupted by yours.
    ///
    /// scholar-server (read-only) uses this entry point. Anything
    /// that intends to write must use [`Database::open_writer`]
    /// instead.
    ///
    /// This automatically performs WAL recovery if needed.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Open FileBackend (which handles WAL recovery internally)
        let backend = FileBackend::open(&path_str)?;
        let engine = Engine::open_or_create(backend)?;

        Ok(Self {
            engine: Arc::new(engine),
            path: path_str,
            _writer_lock: None,
        })
    }

    /// Open or create a database **as the exclusive writer**.
    ///
    /// Acquires a non-blocking `flock(LOCK_EX)` on
    /// `<path>/writer.lock`. If a peer process already holds the
    /// lock, returns an error immediately — no waiting, no race.
    /// The lock is released when this `Database` is dropped (or
    /// when the process dies; the OS releases all flocks at exit).
    ///
    /// Read-only Databases (`Database::open`) are unaffected;
    /// multiple readers + at most one writer can coexist safely.
    ///
    /// **scholar-ingestd uses this entry point.** Two ingestd
    /// processes cannot both run against the same DB simultaneously
    /// — the second one fails fast at this method.
    ///
    /// CoW MVCC Phase 8. See `docs/joule-db/cow-mvcc-design.md`.
    pub fn open_writer<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        use fs2::FileExt;
        use std::fs::{self, OpenOptions};
        use std::path::Path as StdPath;

        let path_str = path.as_ref().to_string_lossy().to_string();

        // Ensure the database directory exists so we can place the
        // lockfile inside it. FileBackend::open does this too, but
        // the lockfile must be created first.
        fs::create_dir_all(&path_str).map_err(|e| {
            Error::Storage(joule_db_core::error::StorageError::Backend(format!(
                "Failed to create database dir {}: {}",
                path_str, e
            )))
        })?;

        let lock_path = StdPath::new(&path_str).join("writer.lock");
        let lock_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| {
                Error::Storage(joule_db_core::error::StorageError::Backend(format!(
                    "Failed to open writer lockfile {}: {}",
                    lock_path.display(),
                    e
                )))
            })?;

        // Non-blocking exclusive lock. fs2 surfaces a distinct
        // `WouldBlock` error when the lock is held; map it to a
        // human-friendly message that names the path so operators
        // can find the offender.
        lock_file.try_lock_exclusive().map_err(|e| {
            Error::Storage(joule_db_core::error::StorageError::Backend(format!(
                "writer lock {} is held by another process — \
                 only one writer may operate on a JouleDB at a time. \
                 If a previous writer crashed, the OS should have \
                 released the lock automatically; if you see this \
                 error persistently, check `lsof {}` for the holder. \
                 Underlying: {}",
                lock_path.display(),
                lock_path.display(),
                e,
            )))
        })?;

        // Stamp the lockfile with our pid so operators can `cat` it
        // to identify the writer. Best-effort; if we crash before
        // the write, the lock is still held.
        {
            use std::io::Write;
            let mut f = &lock_file;
            let _ = f.write_all(format!("{}\n", std::process::id()).as_bytes());
            let _ = f.sync_all();
        }

        // Now do the normal open dance — same as `open` except we
        // hold `_writer_lock`.
        let backend = FileBackend::open(&path_str)?;
        let engine = Engine::open_or_create(backend)?;

        Ok(Self {
            engine: Arc::new(engine),
            path: path_str,
            _writer_lock: Some(lock_file),
        })
    }

    /// Create a new database (fails if already exists)
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Check if database already exists
        let meta_path = std::path::Path::new(&path_str).join("meta.wdb");
        if meta_path.exists() {
            return Err(Error::Storage(joule_db_core::error::StorageError::Backend(
                "Database already exists".to_string(),
            )));
        }

        let backend = FileBackend::open(&path_str)?;
        let engine = Engine::new(backend)?;

        Ok(Self {
            engine: Arc::new(engine),
            path: path_str,
            // `create` is the legacy explicit-create entry point;
            // it doesn't take the writer lock either. Callers
            // intending to write should follow up with
            // `open_writer`. (In practice, scholar uses
            // `open_writer` directly; `create` is mainly for
            // tests.)
            _writer_lock: None,
        })
    }

    /// Get a value by key.
    ///
    /// If the key has a TTL and it has expired, returns `None` and lazily
    /// deletes the key and its TTL metadata.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        // Check TTL first
        if self.is_expired(key)? {
            // Lazy deletion of expired key
            let _ = self.engine.delete(key);
            let _ = self.engine.delete(&ttl_key(key));
            return Ok(None);
        }
        self.engine.get(key)
    }

    /// Get a value without checking TTL (raw access).
    pub fn get_raw(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        self.engine.get(key)
    }

    /// Set a value by key (no TTL — permanent until deleted).
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<(), Error> {
        // Remove any existing TTL when putting without TTL
        let tk = ttl_key(key);
        if self.engine.get(&tk)?.is_some() {
            let _ = self.engine.delete(&tk);
        }
        self.engine.put(key, value)
    }

    /// Set a value with a TTL (time-to-live) in seconds.
    ///
    /// The key will automatically expire after `ttl_seconds` seconds.
    /// Expired keys are lazily deleted on next access, or can be cleaned
    /// up in bulk via `cleanup_expired()`.
    pub fn put_with_ttl(&self, key: &[u8], value: &[u8], ttl_seconds: u64) -> Result<(), Error> {
        let expiry = now_secs() + ttl_seconds;
        let expiry_bytes = expiry.to_be_bytes();
        // Write value first, then TTL metadata. If a crash occurs between the
        // two writes, the value exists without an expiry (safe: it becomes
        // permanent) rather than an expiry existing without a value.
        self.engine.put(key, value)?;
        self.engine.put(&ttl_key(key), &expiry_bytes)
    }

    /// Get the remaining TTL for a key in seconds.
    /// Returns `None` if the key has no TTL or doesn't exist.
    pub fn ttl(&self, key: &[u8]) -> Result<Option<u64>, Error> {
        match self.engine.get(&ttl_key(key))? {
            Some(expiry_bytes) if expiry_bytes.len() == 8 => {
                let expiry = u64::from_be_bytes(slice_to_array(&expiry_bytes)?);
                let now = now_secs();
                if expiry > now {
                    Ok(Some(expiry - now))
                } else {
                    Ok(Some(0)) // expired
                }
            }
            _ => Ok(None),
        }
    }

    /// Remove TTL from a key (make it permanent).
    pub fn persist(&self, key: &[u8]) -> Result<bool, Error> {
        self.engine.delete(&ttl_key(key))
    }

    /// Delete a key and its TTL metadata.
    pub fn delete(&self, key: &[u8]) -> Result<bool, Error> {
        let _ = self.engine.delete(&ttl_key(key));
        self.engine.delete(key)
    }

    /// Check if a key exists
    pub fn contains(&self, key: &[u8]) -> Result<bool, Error> {
        Ok(self.get(key)?.is_some())
    }

    /// Scan all keys with a given prefix, returning (key, value) pairs.
    ///
    /// Expired keys (past their TTL) are filtered out and lazily deleted.
    pub fn prefix_scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, Error> {
        let iter = self.engine.prefix_scan(prefix)?;
        let mut results = Vec::new();
        for entry in iter {
            let entry = entry?;
            // Skip TTL metadata keys
            if entry.key.starts_with(TTL_PREFIX) {
                continue;
            }
            // Skip expired keys (lazy deletion)
            if self.is_expired(&entry.key)? {
                let _ = self.engine.delete(&entry.key);
                let _ = self.engine.delete(&ttl_key(&entry.key));
                continue;
            }
            results.push((entry.key, entry.value));
        }
        Ok(results)
    }

    /// Sync to disk (ensures durability)
    pub fn sync(&self) -> Result<(), Error> {
        self.engine.sync()
    }

    /// Get the database path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get cache size (number of nodes in memory)
    pub fn cache_size(&self) -> usize {
        self.engine.cache_size()
    }

    /// Count all keys in the database
    ///
    /// Note: This performs a full scan and may be slow for large databases.
    /// Use sparingly, preferably for admin/monitoring purposes only.
    pub fn key_count(&self) -> Result<u64, Error> {
        use joule_db_core::index::ScanDirection;
        let mut iter = self.engine.scan(ScanDirection::Forward)?;
        let mut count = 0u64;
        while iter.next().is_some() {
            count += 1;
        }
        Ok(count)
    }

    /// Collect all key-value pairs from the database
    ///
    /// Returns a vector of (key, value) pairs in sorted order.
    /// Note: This loads all data into memory - use with caution for large databases.
    pub fn collect_all(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>, Error> {
        use joule_db_core::index::ScanDirection;
        let mut iter = self.engine.scan(ScanDirection::Forward)?;
        let mut results = Vec::new();
        while let Some(result) = iter.next() {
            let entry = result.map_err(|e| Error::Index(e))?;
            results.push((entry.key, entry.value));
        }
        Ok(results)
    }

    /// Get database statistics
    pub fn stats(&self) -> DatabaseStats {
        DatabaseStats {
            cache_size: self.engine.cache_size(),
            active_latches: self.engine.active_latches(),
        }
    }

    /// Access the underlying engine
    pub fn engine(&self) -> Arc<Engine> {
        Arc::clone(&self.engine)
    }

    /// **CoW MVCC Phase 3 + 4.** Open a read-consistent snapshot at
    /// the engine's most recently committed `(committed_root,
    /// committed_version)`. While the returned [`DbSnapshot`] is
    /// alive:
    ///
    /// - **Phase 3 (in-process):** the engine's live-snapshot counter
    ///   is incremented; deferred `free_page` calls accumulate in the
    ///   engine's pending queue.
    /// - **Phase 4 (cross-process):** a lockfile is registered at
    ///   `<db_path>/snapshots/<pid>.<seq>.lock` and held under an
    ///   exclusive `flock`. Peer processes that scan the directory
    ///   see the lockfile and defer their own frees too.
    ///
    /// On drop: the engine's counter decrements; if it reaches 0 AND
    /// no peer process holds a snapshot, the pending queue drains.
    /// The lockfile is unlinked atomically with the lock release.
    ///
    /// See `docs/joule-db/cow-mvcc-design.md`.
    pub fn open_snapshot(&self) -> Result<DbSnapshot, Error> {
        // Capture the engine's current committed_version BEFORE
        // creating the lockfile so the file's contents reflect the
        // exact version the snapshot pins. Tiny race window: the
        // engine could commit between this read and Snapshot::open's
        // own read of current_committed_meta. That's fine — the
        // version we register in the lockfile is the *minimum* a
        // peer might infer about us, and Snapshot::open captures the
        // actual reading version atomically inside the engine.
        let pinned_version = self.engine.current_committed_meta().committed_version;

        let lock = snapshot_registry::register(
            std::path::Path::new(&self.path),
            pinned_version,
        )
        .map_err(Error::Storage)?;
        let inner = joule_db_core::Snapshot::open(Arc::clone(&self.engine));
        Ok(DbSnapshot { inner, _lock: lock })
    }
}

/// Database-level snapshot wrapping the in-process [`joule_db_core::Snapshot`]
/// with a cross-process lockfile that signals "snapshot live in pid X
/// at version V" to peer processes sharing this database.
///
/// **Drop order matters.** The inner Snapshot drops first — releasing
/// the engine's in-process counter and (if last) draining pending
/// frees. The lockfile drops second — only after our process can no
/// longer issue reads against the snapshot's pages. Peer processes
/// that scan `<db>/snapshots/` between these two events will still
/// see our lockfile, so they'll continue deferring frees until
/// after our reads are guaranteed to be over.
///
/// CoW MVCC Phase 4. See `docs/joule-db/cow-mvcc-design.md`.
pub struct DbSnapshot {
    /// The in-process engine snapshot — produces all the actual reads.
    /// Drops first by struct-field declaration order.
    inner: joule_db_core::Snapshot,
    /// Cross-process lockfile. Drops second; releases the file lock
    /// and unlinks the file from `<db>/snapshots/`.
    _lock: snapshot_registry::SnapshotLockFile,
}

impl DbSnapshot {
    /// Look up a key from this snapshot's pinned root.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        self.inner.get(key)
    }

    /// Root page id captured at snapshot construction.
    pub fn root(&self) -> joule_db_core::storage::PageId {
        self.inner.root()
    }

    /// Committed version captured at snapshot construction.
    pub fn version(&self) -> u64 {
        self.inner.version()
    }

    /// Re-capture the engine's current committed view, picking up
    /// commits from **peer processes** by re-reading meta.wdb from
    /// disk. Returns the new `(root, version)` after refresh.
    ///
    /// scholar-server uses this in a periodic background task to
    /// observe scholar-ingestd's commits without restarting; see
    /// `docs/joule-db/cow-mvcc-design.md` §6.
    ///
    /// The cross-process lockfile remains in place — peer processes
    /// still see we're holding a snapshot.
    pub fn refresh(&mut self) -> Result<(), Error> {
        self.inner.refresh_from_backend()
    }

    /// Range-scan keys in `[start, end)` from this snapshot's pinned root.
    pub fn range(
        &self,
        start: joule_db_core::index::Bound<&[u8]>,
        end: joule_db_core::index::Bound<&[u8]>,
        direction: joule_db_core::index::ScanDirection,
    ) -> Result<joule_db_core::engine::BTreeRangeIterator<'_>, Error> {
        self.inner.range(start, end, direction)
    }

    /// Full scan from this snapshot's pinned root.
    pub fn scan(
        &self,
        direction: joule_db_core::index::ScanDirection,
    ) -> Result<joule_db_core::engine::BTreeRangeIterator<'_>, Error> {
        self.inner.scan(direction)
    }

    /// Prefix scan from this snapshot's pinned root.
    pub fn prefix_scan(
        &self,
        prefix: &[u8],
    ) -> Result<joule_db_core::engine::BTreeRangeIterator<'_>, Error> {
        self.inner.prefix_scan(prefix)
    }
}

impl Database {
    /// Execute an atomic transaction (multi-key update)
    ///
    /// This holds a write lock for the duration of the closure.
    /// Use this for operations that require atomicity across multiple keys.
    pub fn transactional_update<F, T>(&self, f: F) -> Result<T, Error>
    where
        F: FnOnce(&mut WriteTransaction<'_>) -> Result<T, Error>,
    {
        self.engine.write_transaction(f)
    }

    // --- TTL helpers ---

    /// Check if a key has expired. Returns false if no TTL is set.
    fn is_expired(&self, key: &[u8]) -> Result<bool, Error> {
        match self.engine.get(&ttl_key(key))? {
            Some(expiry_bytes) if expiry_bytes.len() == 8 => {
                let expiry = u64::from_be_bytes(slice_to_array(&expiry_bytes)?);
                Ok(now_secs() >= expiry)
            }
            _ => Ok(false),
        }
    }

    /// Scan and delete all expired keys.
    ///
    /// Returns the number of keys cleaned up. Call periodically
    /// (e.g., every 60 seconds) to reclaim space from expired keys.
    pub fn cleanup_expired(&self) -> Result<usize, Error> {
        use joule_db_core::index::{Bound, ScanDirection};

        let now = now_secs();
        // End bound: TTL_PREFIX with last byte incremented
        let mut end_prefix = TTL_PREFIX.to_vec();
        if let Some(last) = end_prefix.last_mut() {
            *last += 1;
        }

        let mut iter = self.engine.range(
            Bound::Included(TTL_PREFIX),
            Bound::Excluded(end_prefix.as_slice()),
            ScanDirection::Forward,
        )?;
        let mut expired_keys = Vec::new();

        while let Some(result) = iter.next() {
            let entry = result.map_err(Error::Index)?;
            if entry.value.len() == 8 {
                let expiry = u64::from_be_bytes(slice_to_array(&entry.value)?);
                if now >= expiry {
                    // Extract original key from TTL key
                    let original_key = entry.key[TTL_PREFIX.len()..].to_vec();
                    expired_keys.push(original_key);
                }
            }
        }
        drop(iter);

        let count = expired_keys.len();
        for key in expired_keys {
            let _ = self.engine.delete(&key);
            let _ = self.engine.delete(&ttl_key(&key));
        }

        Ok(count)
    }
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DatabaseStats {
    /// Number of nodes in the buffer cache
    pub cache_size: usize,
    /// Number of active page latches
    pub active_latches: usize,
}

// Simplified API without complex transaction support for now
// Transaction support through FileBackend is complex due to ownership

// Re-export StorageStats for backward compatibility
pub use joule_db_core::storage::StorageStats;

// Re-export LSM types
pub use lsm::{LsmConfig, LsmEngine};

/// LSM-Tree backed database instance.
///
/// Write-optimized alternative to the B-tree `Database`. Best for
/// write-heavy workloads: time-series, logging, IoT telemetry.
pub struct LsmDatabase {
    engine: lsm::LsmEngine,
    path: String,
}

impl LsmDatabase {
    /// Open or create an LSM database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        Self::open_with_config(path, LsmConfig::default())
    }

    /// Open with custom configuration.
    pub fn open_with_config<P: AsRef<Path>>(path: P, config: LsmConfig) -> Result<Self, Error> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let engine = lsm::LsmEngine::open(path.as_ref(), config).map_err(|e| {
            Error::Storage(joule_db_core::error::StorageError::Backend(e.to_string()))
        })?;
        Ok(Self {
            engine,
            path: path_str,
        })
    }

    /// Get a value by key.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        self.engine
            .get(key)
            .map_err(|e| Error::Storage(joule_db_core::error::StorageError::Backend(e.to_string())))
    }

    /// Set a value by key.
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<(), Error> {
        self.engine
            .put(key.to_vec(), value.to_vec())
            .map_err(|e| Error::Storage(joule_db_core::error::StorageError::Backend(e.to_string())))
    }

    /// Delete a key.
    pub fn delete(&mut self, key: &[u8]) -> Result<bool, Error> {
        // Check if key exists first
        let existed = self.get(key)?.is_some();
        self.engine.delete(key.to_vec()).map_err(|e| {
            Error::Storage(joule_db_core::error::StorageError::Backend(e.to_string()))
        })?;
        Ok(existed)
    }

    /// Check if a key exists.
    pub fn contains(&self, key: &[u8]) -> Result<bool, Error> {
        Ok(self.get(key)?.is_some())
    }

    /// Range scan over keys in [start, end] (inclusive).
    pub fn range(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, Error> {
        self.engine
            .range(start, end)
            .map_err(|e| Error::Storage(joule_db_core::error::StorageError::Backend(e.to_string())))
    }

    /// Sync: flush memtable and compact.
    pub fn sync(&mut self) -> Result<(), Error> {
        self.engine
            .sync()
            .map_err(|e| Error::Storage(joule_db_core::error::StorageError::Backend(e.to_string())))
    }

    /// Flush the active memtable to disk.
    pub fn flush(&mut self) -> Result<(), Error> {
        self.engine
            .flush()
            .map_err(|e| Error::Storage(joule_db_core::error::StorageError::Backend(e.to_string())))
    }

    /// Run compaction if needed.
    pub fn compact(&mut self) -> Result<bool, Error> {
        self.engine
            .compact()
            .map_err(|e| Error::Storage(joule_db_core::error::StorageError::Backend(e.to_string())))
    }

    /// Get the database path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Number of entries in the active memtable.
    pub fn memtable_size(&self) -> usize {
        self.engine.memtable_size()
    }

    /// Total SSTable count across all levels.
    pub fn total_sstables(&self) -> usize {
        self.engine.total_sstables()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use joule_db_core::storage::PageId;
    use tempfile::TempDir;

    #[test]
    fn test_database_open_create() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Create database and write data
        {
            let db = Database::open(&path).unwrap();
            db.put(b"key", b"value").unwrap();
            db.sync().unwrap();
            // Explicit drop before reopen
            drop(db);
        }

        // Small delay to ensure file system has flushed
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Reopen database and verify data persisted
        {
            let db = Database::open(&path).unwrap();
            let value = db.get(b"key").unwrap();
            assert_eq!(value, Some(b"value".to_vec()));
        }
    }

    #[test]
    fn test_database_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        // Put and get
        db.put(b"key1", b"value1").unwrap();
        assert_eq!(db.get(b"key1").unwrap(), Some(b"value1".to_vec()));

        // Contains
        assert!(db.contains(b"key1").unwrap());
        assert!(!db.contains(b"nonexistent").unwrap());

        // Delete
        assert!(db.delete(b"key1").unwrap());
        assert!(!db.delete(b"key1").unwrap()); // Already deleted
        assert_eq!(db.get(b"key1").unwrap(), None);
    }

    /// **CoW MVCC Phase 4.5.** Cross-process snapshot deferral: a
    /// lockfile authored by a "peer process" (faked here by writing
    /// directly under a synthetic peer pid + holding the lock from
    /// inside a thread) must cause the writer's `defer_free_page`
    /// path to queue freed pages instead of releasing them.
    ///
    /// Once the peer's lockfile is released + unlinked, a subsequent
    /// `Engine::sync` opportunistically drains the queue and the
    /// pages become reusable.
    ///
    /// Mechanism under test:
    /// 1. Writer process performs deletes that trigger merges.
    /// 2. With a peer lockfile present, merges queue freed page ids
    ///    in `pending_free_pages` (because `any_external_snapshots_live
    ///    -> true`).
    /// 3. Drop the peer lockfile, run `sync` again, verify the queue
    ///    drains.
    #[test]
    fn external_snapshot_defers_writer_frees_until_peer_releases() {
        use fs2::FileExt;
        use std::fs::OpenOptions;

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path();
        let db = Database::open(db_path).unwrap();

        // Seed data with byte-fat values (2KB) that previously
        // exercised the merge-overflow bug. The byte-size-based
        // rebalance now refuses to merge pages that would exceed
        // page size, so this is safe to run at scale. Many leaves
        // → many merges in the delete phase below → exercises the
        // cross-process deferral semantic against real merge churn.
        let big_val = vec![b'X'; 2048];
        for i in 0..200 {
            db.put(format!("k{:05}", i).as_bytes(), &big_val).unwrap();
        }
        db.sync().unwrap();

        // Synthesise a "peer process" lockfile: write directly under a
        // pid that is NOT our own (so the registry scan treats it as
        // external rather than skipping it). Take an exclusive flock
        // from this thread to keep `try_lock_exclusive` failing.
        let snapshots_dir =
            db_path.join(crate::snapshot_registry::SNAPSHOTS_DIR_NAME);
        std::fs::create_dir_all(&snapshots_dir).unwrap();
        let fake_peer_pid = std::process::id().wrapping_add(99_991); // ≠ our pid
        let peer_lock_path =
            snapshots_dir.join(format!("{}.0.lock", fake_peer_pid));
        let peer_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&peer_lock_path)
            .unwrap();
        peer_file.try_lock_exclusive().expect("peer lock acquired");

        // Sanity: the registry must report the peer as live.
        assert!(
            crate::snapshot_registry::any_external_live(db_path),
            "peer lockfile should be detected as live"
        );

        // Trigger merges via deletes. defer_free_page queues onto
        // pending_free_pages (cross-process snapshot detected by
        // the FileBackend). Byte-fat values stress the byte-size
        // merge feasibility check — merges that would overflow the
        // page get skipped, leaving safely-underfull leaves.
        for i in 0..190 {
            db.delete(format!("k{:05}", i).as_bytes()).unwrap();
        }
        db.sync().unwrap();

        // The pending queue should be non-empty: peer is live so
        // sync's opportunistic drain saw `any_external_snapshots_live`
        // and refused to release. Inspect the engine's pending list
        // length via a probing helper.
        let pending_before_release = db.engine().pending_free_count();
        assert!(
            pending_before_release > 0,
            "pending_free_pages should be non-empty while peer holds lock; \
             saw {}",
            pending_before_release
        );

        // Release the peer's lock + unlink. Now the registry says
        // no peers are live.
        FileExt::unlock(&peer_file).unwrap();
        drop(peer_file);
        let _ = std::fs::remove_file(&peer_lock_path);
        assert!(
            !crate::snapshot_registry::any_external_live(db_path),
            "post-release: registry must not report any peers live"
        );

        // Sync again — this time `try_drain_pending_frees` succeeds
        // because both in-process count is 0 and no peers are live.
        db.sync().unwrap();

        let pending_after_release = db.engine().pending_free_count();
        assert_eq!(
            pending_after_release, 0,
            "pending_free_pages should drain to 0 once the peer's \
             lockfile is gone; saw {}",
            pending_after_release
        );
    }

    /// **CoW MVCC Phase 3.5.** Snapshot via the public Database API
    /// against a real FileBackend. Proves the v1 meta.wdb +
    /// snapshot machinery composes correctly: a snapshot taken
    /// after sync 1 sees only sync-1 data even after sync 2 + sync 3
    /// land on disk. Refresh advances the snapshot through the v1
    /// records.
    #[test]
    fn database_snapshot_isolates_reads_from_concurrent_writes() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        // Sync 1: seed three keys.
        db.put(b"alpha", b"v1-A").unwrap();
        db.put(b"beta", b"v1-B").unwrap();
        db.sync().unwrap();

        let snap1 = db.open_snapshot().unwrap();
        assert_eq!(snap1.version(), 1);
        assert_eq!(snap1.get(b"alpha").unwrap().as_deref(), Some(&b"v1-A"[..]));
        assert_eq!(snap1.get(b"gamma").unwrap(), None);

        // Sync 2: overwrite alpha + add gamma.
        db.put(b"alpha", b"v2-A").unwrap();
        db.put(b"gamma", b"v2-G").unwrap();
        db.sync().unwrap();

        // The pinned snapshot still observes the sync-1 state.
        assert_eq!(
            snap1.get(b"alpha").unwrap().as_deref(),
            Some(&b"v1-A"[..]),
            "snap1 must NOT see sync-2 alpha overwrite"
        );
        assert_eq!(snap1.get(b"gamma").unwrap(), None);

        // Database default reads see the new state.
        assert_eq!(db.get(b"alpha").unwrap().as_deref(), Some(&b"v2-A"[..]));
        assert_eq!(db.get(b"gamma").unwrap().as_deref(), Some(&b"v2-G"[..]));

        // Open a second snapshot — it captures sync 2's view, not sync 1.
        let snap2 = db.open_snapshot().unwrap();
        assert_eq!(snap2.version(), 2);
        assert_eq!(snap2.get(b"alpha").unwrap().as_deref(), Some(&b"v2-A"[..]));

        // Sync 3 advances the writer; both snapshots stay pinned.
        db.put(b"delta", b"v3-D").unwrap();
        db.sync().unwrap();
        assert_eq!(snap1.get(b"delta").unwrap(), None);
        assert_eq!(snap2.get(b"delta").unwrap(), None);
        assert_eq!(db.get(b"delta").unwrap().as_deref(), Some(&b"v3-D"[..]));

        // Drop snap1, refresh snap2 — snap2 should now see v3.
        drop(snap1);
        let mut snap2 = snap2;
        snap2.refresh().unwrap();
        assert_eq!(snap2.version(), 3);
        assert_eq!(snap2.get(b"delta").unwrap().as_deref(), Some(&b"v3-D"[..]));
    }

    /// **CoW MVCC Phase 8.** A second `open_writer` against the
    /// same path must fail fast while the first holds the lock —
    /// not block, not silently succeed. After the first drops, a
    /// new `open_writer` succeeds.
    ///
    /// Regression test for the 2026-04-29 9-hour dual-writer hang
    /// where two scholar-ingestd processes (manual ROR + supervisor
    /// daemon tick) both opened the same DB and corrupted each
    /// other.
    #[test]
    fn open_writer_excludes_concurrent_writer() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        let writer_a = Database::open_writer(&path).expect("first writer should open");

        // Second open_writer should fail fast (non-blocking).
        let writer_b = Database::open_writer(&path);
        match writer_b {
            Ok(_) => panic!(
                "second open_writer should have failed while writer A holds the lock"
            ),
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("writer lock") || msg.contains("WouldBlock"),
                    "error should name the writer lock contention; got: {msg}"
                );
            }
        }

        // Drop A; B should now succeed.
        drop(writer_a);
        let writer_c = Database::open_writer(&path)
            .expect("after writer A drops, writer C should be able to open");
        // C should be writable.
        writer_c.put(b"after_drop", b"ok").unwrap();
        assert_eq!(
            writer_c.get(b"after_drop").unwrap().as_deref(),
            Some(&b"ok"[..])
        );
    }

    /// **Reader compatibility**: `Database::open` must NOT take
    /// the writer lock; readers should always succeed even while
    /// a writer is mid-flight. (scholar-server stays on `open`
    /// after the cutover so the supervisor's daemon ingest tick
    /// can acquire the writer lock concurrently.)
    #[test]
    fn open_reader_does_not_block_or_block_writer() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Seed initial state via a writer, then drop it.
        {
            let w = Database::open_writer(&path).unwrap();
            w.put(b"seed", b"v1").unwrap();
            w.sync().unwrap();
        }

        // Open a reader. Then open a writer concurrently — the
        // reader's existence must not block the writer.
        let reader = Database::open(&path).unwrap();
        let writer = Database::open_writer(&path).expect(
            "writer must be able to open while a reader is alive",
        );

        // Both can coexist. Writer commits new data; reader sees
        // the seed (its snapshot is from the open moment) until
        // refreshed via DbSnapshot::refresh.
        writer.put(b"new_key", b"v2").unwrap();
        writer.sync().unwrap();

        assert_eq!(reader.get(b"seed").unwrap().as_deref(), Some(&b"v1"[..]));

        // A second reader can open even while the writer is alive
        // (no writer lock contention for readers).
        let reader2 = Database::open(&path).unwrap();
        assert_eq!(reader2.get(b"seed").unwrap().as_deref(), Some(&b"v1"[..]));
    }

    /// **CoW MVCC Phase 2.5.** Round-trip the v1 `meta.wdb` record:
    /// open, do some commits, close, reopen — the engine must
    /// recover both the root pointer and the `committed_version` from
    /// the atomically-renamed meta.wdb instead of the legacy page-1
    /// in-place record.
    ///
    /// Property: every `db.sync()` increments `committed_version`,
    /// and the post-reopen version matches the pre-close version.
    #[test]
    fn committed_meta_round_trips_through_reopen() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        let pre_close_version: u64;
        let pre_close_root: PageId;

        {
            let db = Database::open(&path).unwrap();

            // Initial state: a fresh DB has committed_version 0 (no
            // sync has run yet).
            let initial = db.engine().current_committed_meta();
            assert_eq!(initial.committed_version, 0);

            // Three sync rounds; each must bump the version monotonically.
            for batch in 0..3 {
                for i in 0..10 {
                    let key = format!("k_{}_{}", batch, i);
                    let val = format!("v_{}_{}", batch, i);
                    db.put(key.as_bytes(), val.as_bytes()).unwrap();
                }
                db.sync().unwrap();
            }

            let after = db.engine().current_committed_meta();
            assert_eq!(
                after.committed_version, 3,
                "expected 3 sync calls to produce committed_version=3"
            );
            assert!(
                after.committed_root != 0,
                "committed_root must be a real page"
            );
            pre_close_version = after.committed_version;
            pre_close_root = after.committed_root;
        }

        // Reopen — Engine must read root + version from the
        // atomically-committed meta.wdb v1 record, not the legacy
        // page-1 cache.
        {
            let db = Database::open(&path).unwrap();
            let reopened = db.engine().current_committed_meta();
            assert_eq!(
                reopened.committed_version, pre_close_version,
                "committed_version must survive reopen via meta.wdb v1"
            );
            assert_eq!(
                reopened.committed_root, pre_close_root,
                "committed_root must survive reopen via meta.wdb v1"
            );

            // Sanity: data is intact.
            for batch in 0..3 {
                for i in 0..10 {
                    let key = format!("k_{}_{}", batch, i);
                    let expected = format!("v_{}_{}", batch, i);
                    assert_eq!(
                        db.get(key.as_bytes()).unwrap(),
                        Some(expected.into_bytes()),
                    );
                }
            }

            // A new sync after reopen continues the version sequence.
            db.put(b"after_reopen", b"yes").unwrap();
            db.sync().unwrap();
            let after_extra_sync = db.engine().current_committed_meta();
            assert_eq!(after_extra_sync.committed_version, pre_close_version + 1);
        }
    }

    #[test]
    fn test_database_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Write multiple keys
        {
            let db = Database::open(&path).unwrap();
            for i in 0..20 {
                let key = format!("key{:03}", i);
                let value = format!("value{}", i);
                db.put(key.as_bytes(), value.as_bytes()).unwrap();
            }
            db.sync().unwrap();
        }

        // Reopen and verify all keys
        {
            let db = Database::open(&path).unwrap();
            for i in 0..20 {
                let key = format!("key{:03}", i);
                let expected = format!("value{}", i);
                let actual = db.get(key.as_bytes()).unwrap();
                assert_eq!(
                    actual,
                    Some(expected.into_bytes()),
                    "Failed for key {}",
                    key
                );
            }
        }
    }

    #[test]
    fn test_database_many_keys() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        // Insert many keys to trigger B-tree splits
        for i in 0..100 {
            let key = format!("key{:04}", i);
            let value = format!("value{}", i);
            db.put(key.as_bytes(), value.as_bytes()).unwrap();
        }

        // Verify all keys
        for i in 0..100 {
            let key = format!("key{:04}", i);
            let expected = format!("value{}", i);
            assert_eq!(
                db.get(key.as_bytes()).unwrap(),
                Some(expected.into_bytes()),
                "Failed for key {}",
                key
            );
        }
    }

    #[test]
    fn test_database_sync() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        db.put(b"test", b"data").unwrap();
        db.sync().unwrap();

        // Should not panic
        assert_eq!(db.get(b"test").unwrap(), Some(b"data".to_vec()));
    }

    #[test]
    fn test_database_cache_size() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        // Initially should have at least the root node cached
        let initial_size = db.cache_size();
        assert!(initial_size > 0);

        // Add more data
        for i in 0..50 {
            let key = format!("key{}", i);
            db.put(key.as_bytes(), b"value").unwrap();
        }

        // Cache should have grown
        assert!(db.cache_size() >= initial_size);
    }

    #[test]
    fn test_database_key_count() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        // Initially should have 0 keys
        assert_eq!(db.key_count().unwrap(), 0);

        // Add some keys
        for i in 0..25 {
            let key = format!("key{:03}", i);
            db.put(key.as_bytes(), b"value").unwrap();
        }

        // Should have 25 keys
        assert_eq!(db.key_count().unwrap(), 25);

        // Delete some keys
        for i in 0..5 {
            let key = format!("key{:03}", i);
            db.delete(key.as_bytes()).unwrap();
        }

        // Should have 20 keys
        assert_eq!(db.key_count().unwrap(), 20);
    }

    #[test]
    fn test_database_stats() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        // Add some data
        for i in 0..10 {
            let key = format!("key{}", i);
            db.put(key.as_bytes(), b"value").unwrap();
        }

        let stats = db.stats();
        assert!(stats.cache_size > 0);
        // active_latches may be 0 depending on implementation
    }

    #[test]
    fn test_database_collect_all() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        // Add some keys in reverse order
        for i in (0..10).rev() {
            let key = format!("key{:02}", i);
            let value = format!("value{}", i);
            db.put(key.as_bytes(), value.as_bytes()).unwrap();
        }

        // Collect all should return sorted keys
        let all = db.collect_all().unwrap();
        assert_eq!(all.len(), 10);

        // Verify sorted order
        for (i, (key, value)) in all.iter().enumerate() {
            let expected_key = format!("key{:02}", i);
            let expected_value = format!("value{}", i);
            assert_eq!(key, expected_key.as_bytes());
            assert_eq!(value, expected_value.as_bytes());
        }
    }

    // ==================== TTL tests ====================

    #[test]
    fn test_put_with_ttl_get() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        // Put with 3600-second TTL (1 hour)
        db.put_with_ttl(b"key1", b"val1", 3600).unwrap();

        // Should be readable immediately
        let val = db.get(b"key1").unwrap();
        assert_eq!(val, Some(b"val1".to_vec()));
    }

    #[test]
    fn test_ttl_returns_remaining_seconds() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        db.put_with_ttl(b"mykey", b"myval", 3600).unwrap();

        // TTL should be approximately 3600 (allow small delta for test execution)
        let remaining = db.ttl(b"mykey").unwrap();
        assert!(remaining.is_some());
        let r = remaining.unwrap();
        assert!(r >= 3590 && r <= 3600, "TTL was {}", r);
    }

    #[test]
    fn test_ttl_none_for_permanent_key() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        db.put(b"perm", b"val").unwrap();

        // Permanent key should have no TTL
        let remaining = db.ttl(b"perm").unwrap();
        assert!(remaining.is_none());
    }

    #[test]
    fn test_persist_removes_ttl() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        db.put_with_ttl(b"temp", b"val", 100).unwrap();
        assert!(db.ttl(b"temp").unwrap().is_some());

        // Persist should remove the TTL
        db.persist(b"temp").unwrap();
        assert!(db.ttl(b"temp").unwrap().is_none());

        // Value should still be readable
        assert_eq!(db.get(b"temp").unwrap(), Some(b"val".to_vec()));
    }

    #[test]
    fn test_delete_removes_ttl() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        db.put_with_ttl(b"key", b"val", 300).unwrap();
        db.delete(b"key").unwrap();

        // Both value and TTL should be gone
        assert_eq!(db.get(b"key").unwrap(), None);
        assert!(db.ttl(b"key").unwrap().is_none());
    }

    #[test]
    fn test_ttl_expired_key_returns_none() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        // Put with TTL=0 (immediately expired)
        // We need to put a TTL in the past. The internal format is Unix timestamp.
        // put_with_ttl adds now_secs() + ttl, so TTL=0 means it expires at now_secs().
        // Since get() checks expiry >= now, TTL=0 will still be valid for a moment.
        // Instead, write TTL metadata directly with an expired timestamp.
        db.put(b"key", b"val").unwrap();
        let ttl_key = format!("__ttl__::key");
        // Write an expired timestamp (1 second in the past)
        let expired_ts = now_secs().saturating_sub(1);
        db.engine
            .put(ttl_key.as_bytes(), &expired_ts.to_be_bytes())
            .unwrap();

        // get() should return None (lazy deletion)
        let val = db.get(b"key").unwrap();
        assert_eq!(val, None);
    }

    #[test]
    fn test_cleanup_expired() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        // Insert some keys with immediate expiry
        for i in 0..5 {
            let key = format!("exp_{}", i);
            db.put(key.as_bytes(), b"val").unwrap();
            let ttl_key = format!("__ttl__::{}", key);
            let expired_ts = now_secs().saturating_sub(1);
            db.engine
                .put(ttl_key.as_bytes(), &expired_ts.to_be_bytes())
                .unwrap();
        }
        // Also insert a non-expired key
        db.put_with_ttl(b"alive", b"yes", 3600).unwrap();

        let cleaned = db.cleanup_expired().unwrap();
        assert_eq!(cleaned, 5);

        // Expired keys should be gone
        for i in 0..5 {
            let key = format!("exp_{}", i);
            assert_eq!(db.get(key.as_bytes()).unwrap(), None);
        }
        // Non-expired key should still be there
        assert_eq!(db.get(b"alive").unwrap(), Some(b"yes".to_vec()));
    }

    #[test]
    fn test_put_overwrites_ttl() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        db.put_with_ttl(b"key", b"val1", 100).unwrap();
        assert!(db.ttl(b"key").unwrap().is_some());

        // Regular put should remove the TTL
        db.put(b"key", b"val2").unwrap();
        assert!(db.ttl(b"key").unwrap().is_none());
        assert_eq!(db.get(b"key").unwrap(), Some(b"val2".to_vec()));
    }

    #[test]
    fn test_put_with_ttl_updates_ttl() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        db.put_with_ttl(b"key", b"val1", 100).unwrap();
        let ttl1 = db.ttl(b"key").unwrap().unwrap();

        db.put_with_ttl(b"key", b"val2", 5000).unwrap();
        let ttl2 = db.ttl(b"key").unwrap().unwrap();

        assert!(
            ttl2 > ttl1,
            "New TTL {} should be greater than old TTL {}",
            ttl2,
            ttl1
        );
        assert_eq!(db.get(b"key").unwrap(), Some(b"val2".to_vec()));
    }

    #[test]
    fn test_get_raw_skips_ttl_check() {
        let temp_dir = TempDir::new().unwrap();
        let db = Database::open(temp_dir.path()).unwrap();

        // Put a key and manually expire it
        db.put(b"key", b"val").unwrap();
        let tk = "__ttl__::key";
        let expired_ts = now_secs().saturating_sub(1);
        db.engine
            .put(tk.as_bytes(), &expired_ts.to_be_bytes())
            .unwrap();

        // get_raw should return the value (skips TTL check)
        assert_eq!(db.get_raw(b"key").unwrap(), Some(b"val".to_vec()));

        // Regular get returns None (expired) and lazy-deletes
        assert_eq!(db.get(b"key").unwrap(), None);
    }
}
