//! Cross-process snapshot registry — Phase 4 of the CoW MVCC refactor.
//!
//! Each `Database::open_snapshot` registers a per-process file lock at
//! `<db_path>/snapshots/<pid>.<seq>.lock`. The file's only purpose is
//! to be `flock`'d for the snapshot's lifetime — its existence + lock
//! state communicates "a snapshot is live in process <pid>" to any
//! other process sharing this database.
//!
//! Why files + flock and not e.g. shared memory:
//!
//! 1. **Survives any peer's crash.** If the peer process dies (kill -9,
//!    panic without unwind, etc.), the OS releases its `flock` and
//!    closes its file descriptor. The next writer to scan the
//!    directory finds the abandoned lockfile, takes the lock itself
//!    (succeeds because no one's holding it), and unlinks — automatic
//!    cleanup of stale entries with zero coordination.
//!
//! 2. **No daemon, no IPC.** Plain filesystem operations.
//!
//! 3. **Portable.** `fs2::FileExt::try_lock_exclusive` works on Linux,
//!    macOS, and Windows.
//!
//! See `docs/joule-db/cow-mvcc-design.md` §4.5.

use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use fs2::FileExt;

use joule_db_core::error::StorageError;

/// Subdirectory under the database root holding per-process snapshot
/// lockfiles. Created lazily on first `register`.
pub const SNAPSHOTS_DIR_NAME: &str = "snapshots";

/// Monotonic per-process sequence number for lockfile names.
/// Combined with `pid` to guarantee a unique path even when the same
/// process opens many snapshots concurrently.
static SNAPSHOT_SEQ: AtomicU64 = AtomicU64::new(0);

/// Owned handle to a snapshot lockfile. While this value lives, the
/// lockfile exists at `<db>/snapshots/<pid>.<seq>.lock` and the OS
/// holds an exclusive `flock` on it.
///
/// Drop releases the lock and unlinks the file. If the process dies
/// without dropping, the OS still releases the lock — the next
/// writer's GC pass picks up and unlinks the stale lockfile.
pub struct SnapshotLockFile {
    /// Holds the OS file descriptor + advisory lock. Closed on Drop,
    /// which releases the flock.
    file: Option<File>,
    /// Path to unlink on Drop.
    path: PathBuf,
}

impl SnapshotLockFile {
    /// Path for testing/debug.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for SnapshotLockFile {
    fn drop(&mut self) {
        // Closing the file releases the flock atomically with respect
        // to other processes scanning the directory. Unlink afterwards
        // — if unlink fails (filesystem race, dir gone, etc.), the
        // file lingers but a future writer's scan will find it
        // unlocked and unlink it. So this Drop never panics.
        drop(self.file.take());
        let _ = fs::remove_file(&self.path);
    }
}

/// Register a new snapshot lockfile under `<db_path>/snapshots/`.
///
/// Creates the snapshots directory on first call. Encodes the
/// snapshot's `committed_version` in the file's contents so future
/// scans can determine the watermark without rejoining process state.
/// Takes a non-blocking exclusive lock — failure is fatal because it
/// implies our chosen `(pid, seq)` path is somehow already locked
/// (which our atomic seq counter is supposed to prevent).
pub fn register(
    db_path: &Path,
    committed_version: u64,
) -> Result<SnapshotLockFile, StorageError> {
    let snapshots_dir = db_path.join(SNAPSHOTS_DIR_NAME);
    fs::create_dir_all(&snapshots_dir).map_err(|e| {
        StorageError::Backend(format!(
            "Failed to create snapshots dir {}: {}",
            snapshots_dir.display(),
            e
        ))
    })?;

    let seq = SNAPSHOT_SEQ.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let lock_path = snapshots_dir.join(format!("{}.{}.lock", pid, seq));

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&lock_path)
        .map_err(|e| {
            StorageError::Backend(format!(
                "Failed to create snapshot lockfile {}: {}",
                lock_path.display(),
                e
            ))
        })?;

    file.try_lock_exclusive().map_err(|e| {
        StorageError::Backend(format!(
            "Failed to acquire snapshot lockfile {}: {}",
            lock_path.display(),
            e
        ))
    })?;

    // Write the snapshot version as a 16-byte hex string. Plain text
    // so an operator can `cat` the file and see what's locking the
    // database. Length-fixed for easy parsing in a future
    // `min_live_version` scan.
    use std::io::Write;
    let mut f = &file;
    f.write_all(format!("{:016x}\n", committed_version).as_bytes())
        .map_err(|e| {
            StorageError::Backend(format!(
                "Failed to write snapshot version to {}: {}",
                lock_path.display(),
                e
            ))
        })?;
    f.sync_all().map_err(|e| {
        StorageError::Backend(format!(
            "Failed to sync snapshot lockfile {}: {}",
            lock_path.display(),
            e
        ))
    })?;

    Ok(SnapshotLockFile {
        file: Some(file),
        path: lock_path,
    })
}

/// Scan `<db_path>/snapshots/` looking for live snapshots in **other
/// processes**. Returns true if any peer process holds a snapshot.
///
/// Logic per file:
/// - If the file's name is prefixed by our own PID, skip it. Our own
///   process tracks its in-process snapshots via the engine's
///   `live_snapshots` counter — this scan exists to detect peers.
/// - Try `try_lock_exclusive`. If it **succeeds**, the previous
///   holder is gone (process died, or never held the lock); take the
///   lock momentarily, drop it, and unlink the stale file. If it
///   **fails**, the file is held by a live peer.
///
/// Returns `Ok(true)` as soon as the first live peer is detected
/// (short-circuit). On I/O errors during the scan, returns
/// `Ok(false)` rather than propagating — the caller should treat a
/// scan failure as "can't prove anyone's live" but a follow-up
/// allocation might fail safely. (We could be stricter, but the
/// scan happens on the writer's hot path; transient errors should
/// not abort writes.)
pub fn any_external_live(db_path: &Path) -> bool {
    let snapshots_dir = db_path.join(SNAPSHOTS_DIR_NAME);
    let entries = match fs::read_dir(&snapshots_dir) {
        Ok(e) => e,
        Err(_) => return false, // dir doesn't exist → no snapshots
    };

    let own_pid_prefix = format!("{}.", std::process::id());

    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if !name.ends_with(".lock") {
            continue;
        }
        if name.starts_with(&own_pid_prefix) {
            // Our own snapshots — tracked by in-process counter.
            continue;
        }

        // Try to take exclusive lock. Open+try_lock combo:
        let file = match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(f) => f,
            Err(_) => continue, // raced with peer's drop, ignore
        };

        match file.try_lock_exclusive() {
            Ok(()) => {
                // Peer is gone — clean up. Best-effort unlink.
                fs2::FileExt::unlock(&file).ok();
                drop(file);
                let _ = fs::remove_file(&path);
            }
            Err(_) => {
                // Peer is alive and holding the lock.
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn register_creates_lockfile() {
        let dir = TempDir::new().unwrap();
        let lock = register(dir.path(), 42).unwrap();
        assert!(lock.path().exists(), "lockfile should be created");
        assert!(
            lock.path()
                .to_string_lossy()
                .contains(SNAPSHOTS_DIR_NAME),
            "lockfile lives under snapshots/"
        );
    }

    #[test]
    fn drop_unlinks_lockfile() {
        let dir = TempDir::new().unwrap();
        let lock = register(dir.path(), 7).unwrap();
        let path = lock.path().to_path_buf();
        assert!(path.exists());
        drop(lock);
        assert!(!path.exists(), "lockfile should be unlinked on Drop");
    }

    #[test]
    fn any_external_live_skips_own_pid() {
        let dir = TempDir::new().unwrap();
        // Our own process registers a snapshot. From OUR perspective
        // (same pid), any_external_live should be false — the registry
        // only reports peers.
        let _own = register(dir.path(), 1).unwrap();
        assert!(
            !any_external_live(dir.path()),
            "own-pid lockfile must not count as external"
        );
    }

    #[test]
    fn any_external_live_returns_false_for_empty_or_missing_dir() {
        let dir = TempDir::new().unwrap();
        assert!(!any_external_live(dir.path()));
    }

    #[test]
    fn stale_lockfile_from_dead_peer_gets_cleaned_up() {
        // Simulate a "dead peer" by writing a lockfile under a fake
        // peer pid but NOT holding any flock on it. The registry's
        // any_external_live scan should successfully `try_lock`,
        // recognise it as stale, and unlink.
        let dir = TempDir::new().unwrap();
        let snapshots_dir = dir.path().join(SNAPSHOTS_DIR_NAME);
        fs::create_dir_all(&snapshots_dir).unwrap();

        // Pick a pid that's deliberately different from our own.
        let fake_peer_pid = std::process::id().wrapping_add(1);
        let stale = snapshots_dir.join(format!("{}.0.lock", fake_peer_pid));
        fs::write(&stale, "stale\n").unwrap();
        assert!(stale.exists());

        // Scan: should clean up the stale entry and report no peers
        // are live.
        let live = any_external_live(dir.path());
        assert!(!live, "stale unlocked file must not be considered live");
        assert!(!stale.exists(), "stale lockfile must be unlinked");
    }
}
