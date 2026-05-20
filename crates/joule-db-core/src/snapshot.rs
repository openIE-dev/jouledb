//! Reader snapshots — Phase 3 of the CoW MVCC refactor.
//!
//! A [`Snapshot`] captures `(committed_root, committed_version)` from
//! the engine atomically at construction time. While the snapshot
//! lives, the writer can keep advancing — its CoW writes allocate
//! fresh page ids and atomically swap the meta record — but the
//! snapshot's own root and the pages reachable from it remain valid:
//!
//! - **No in-place mutation.** Phase 1 ensured every B-tree write
//!   produces fresh page ids; the snapshot's pages are never
//!   overwritten.
//! - **Deferred free.** Phase 3 defers any `free_page` call while a
//!   snapshot is live, so the buffer pool's allocator cannot reuse
//!   the snapshot's pages for new writes.
//!
//! Snapshots are created via [`crate::Engine::open_snapshot`]
//! (constructor on this module) or, for full-database access,
//! `joule_db_local::Database::open_snapshot`.
//!
//! See `docs/joule-db/cow-mvcc-design.md`.

use std::sync::Arc;

use crate::engine::{BTreeRangeIterator, Engine};
use crate::error::Error;
use crate::index::{Bound, ScanDirection};
use crate::storage::PageId;

/// Read-consistent view of a B-tree at a specific committed version.
///
/// Holds an `Arc<Engine>` plus the captured `(root, version)`. The
/// Engine's live-snapshot counter is incremented at construction and
/// decremented on drop; while non-zero it gates the writer's
/// `free_page` calls so this snapshot's pages stay reachable.
///
/// `Snapshot` is `Send + Sync` so it can be handed across threads
/// (e.g. scholar-server passes one to each request handler in a
/// future phase).
pub struct Snapshot {
    engine: Arc<Engine>,
    root: PageId,
    version: u64,
}

impl Snapshot {
    /// Open a new snapshot capturing the engine's current
    /// `(committed_root, committed_version)`.
    ///
    /// Increments the engine's live-snapshot counter — drop the
    /// returned `Snapshot` to release.
    pub fn open(engine: Arc<Engine>) -> Self {
        engine.acquire_snapshot();
        let meta = engine.current_committed_meta();
        Self {
            engine,
            root: meta.committed_root,
            version: meta.committed_version,
        }
    }

    /// Root page id captured at snapshot construction. Stable for
    /// the snapshot's lifetime — does not advance even when the
    /// writer commits.
    pub fn root(&self) -> PageId {
        self.root
    }

    /// Committed version captured at snapshot construction.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Look up a key from this snapshot's root. Reads bypass the
    /// engine's mutable `root_page_id` and traverse only pages
    /// reachable from `self.root`, all of which are guaranteed to
    /// be valid for the lifetime of this `Snapshot`.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        self.engine.get_at_root(self.root, key)
    }

    /// Range-scan keys in `[start, end)` from this snapshot's pinned
    /// root. Same lifetime guarantees as `get`.
    pub fn range(
        &self,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        direction: ScanDirection,
    ) -> Result<BTreeRangeIterator<'_>, Error> {
        self.engine.range_at_root(self.root, start, end, direction)
    }

    /// Full scan from this snapshot's pinned root.
    pub fn scan(&self, direction: ScanDirection) -> Result<BTreeRangeIterator<'_>, Error> {
        self.engine.scan_at_root(self.root, direction)
    }

    /// Prefix scan from this snapshot's pinned root.
    pub fn prefix_scan(&self, prefix: &[u8]) -> Result<BTreeRangeIterator<'_>, Error> {
        self.engine.prefix_scan_at_root(self.root, prefix)
    }

    /// Re-capture the engine's current `(committed_root, committed_version)`,
    /// advancing this snapshot to the most recently committed
    /// **in-process** state. Use [`Self::refresh_from_backend`] if
    /// you need to pick up commits from a peer process.
    ///
    /// **Atomicity:** the engine's `current_committed_meta` is read
    /// once and assigned — there is no window where `(root, version)`
    /// could be drawn from different commits.
    pub fn refresh(&mut self) {
        let meta = self.engine.current_committed_meta();
        self.root = meta.committed_root;
        self.version = meta.committed_version;
    }

    /// **CoW MVCC Phase 6.** Re-read the backend's atomically-
    /// committed meta from disk before refreshing. Picks up commits
    /// made by **peer processes** that the in-memory engine state
    /// would otherwise be unaware of (scholar-server reading
    /// scholar-ingestd's commits is the canonical case).
    ///
    /// In a single-process workload this is equivalent to `refresh`
    /// (slight overhead from the backend disk read).
    pub fn refresh_from_backend(&mut self) -> Result<(), Error> {
        let meta = self.engine.refresh_from_backend()?;
        self.root = meta.committed_root;
        self.version = meta.committed_version;
        Ok(())
    }
}

impl Drop for Snapshot {
    fn drop(&mut self) {
        // Release the engine's snapshot counter. If we were the last
        // live snapshot, the engine drains its pending-free queue —
        // see `Engine::release_snapshot`.
        self.engine.release_snapshot();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::MemoryBackend;

    /// **Phase 3 core invariant.** A snapshot taken at version V
    /// keeps observing V's keys/values even as the writer commits
    /// V+1, V+2, ... Refreshing advances the snapshot to the latest
    /// committed view.
    #[test]
    fn snapshot_pinned_view_survives_concurrent_writes() {
        let backend = MemoryBackend::new();
        let engine = Arc::new(Engine::new(backend).unwrap());

        // Round 1: seed three keys, sync → version 1.
        engine.put(b"alpha", b"v1-A").unwrap();
        engine.put(b"beta", b"v1-B").unwrap();
        engine.put(b"gamma", b"v1-C").unwrap();
        engine.sync().unwrap();

        let snap = Snapshot::open(Arc::clone(&engine));
        assert_eq!(snap.version(), 1, "snapshot should capture v1");
        assert_eq!(snap.get(b"alpha").unwrap().as_deref(), Some(&b"v1-A"[..]));

        // Round 2: writer overwrites + adds. Sync → version 2.
        engine.put(b"alpha", b"v2-A").unwrap();
        engine.put(b"delta", b"v2-D").unwrap();
        engine.sync().unwrap();

        // The snapshot still observes the v1 state.
        assert_eq!(
            snap.get(b"alpha").unwrap().as_deref(),
            Some(&b"v1-A"[..]),
            "snapshot must NOT see writer's v2 overwrite"
        );
        assert_eq!(
            snap.get(b"delta").unwrap(),
            None,
            "snapshot must NOT see writer's v2 insert"
        );
        // The engine's current view does see the new state.
        assert_eq!(
            engine.get(b"alpha").unwrap().as_deref(),
            Some(&b"v2-A"[..])
        );
        assert_eq!(engine.get(b"delta").unwrap().as_deref(), Some(&b"v2-D"[..]));

        // Round 3: another commit.
        engine.put(b"epsilon", b"v3-E").unwrap();
        engine.sync().unwrap();

        // Snapshot still pinned to v1.
        assert_eq!(snap.get(b"epsilon").unwrap(), None);
        assert_eq!(snap.version(), 1);

        // Refresh advances the snapshot to the current committed view.
        let mut snap = snap;
        snap.refresh();
        assert_eq!(snap.version(), 3);
        assert_eq!(snap.get(b"alpha").unwrap().as_deref(), Some(&b"v2-A"[..]));
        assert_eq!(snap.get(b"epsilon").unwrap().as_deref(), Some(&b"v3-E"[..]));
    }

    /// While any snapshot is live, the writer's `defer_free_page`
    /// must NOT actually release pages to the buffer pool — that
    /// would let the allocator reuse them, corrupting the snapshot.
    /// Drop the last snapshot and the deferred frees flush.
    #[test]
    fn deferred_frees_released_only_after_last_snapshot_drops() {
        let backend = MemoryBackend::new();
        let engine = Arc::new(Engine::new(backend).unwrap());

        // Seed enough keys to grow past leaf capacity, then sync.
        for i in 0..50 {
            let k = format!("k{:04}", i);
            let v = format!("v{}", i);
            engine.put(k.as_bytes(), v.as_bytes()).unwrap();
        }
        engine.sync().unwrap();

        let snap = Snapshot::open(Arc::clone(&engine));
        let snap_v = snap.version();

        // Trigger a delete-driven merge by removing many keys —
        // this exercises merge_with_left/right which calls
        // defer_free_page. With a snapshot live, no page should be
        // recycled into the next allocation.
        for i in 0..40 {
            let k = format!("k{:04}", i);
            engine.delete(k.as_bytes()).unwrap();
        }
        engine.sync().unwrap();

        // Snapshot must still be readable for the keys it captured.
        for i in 0..50 {
            let k = format!("k{:04}", i);
            let expected = format!("v{}", i);
            assert_eq!(
                snap.get(k.as_bytes()).unwrap(),
                Some(expected.into_bytes()),
                "snapshot v{} must still see key {} despite writer deletes",
                snap_v,
                k
            );
        }

        // Drop the snapshot — pending frees flush. The engine's
        // current view continues to be correct (no double-free, no
        // panic).
        drop(snap);

        // Sanity: engine post-drop is still functional.
        engine.put(b"after-drop", b"ok").unwrap();
        engine.sync().unwrap();
        assert_eq!(
            engine.get(b"after-drop").unwrap().as_deref(),
            Some(&b"ok"[..])
        );
    }
}
