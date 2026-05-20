//! **CoW MVCC Phase 5 — concurrent reader+writer integration tests.**
//!
//! Phase 1–4 unit tests cover individual mechanics. This integration
//! suite exercises the full MVCC pipeline under threaded contention
//! and validates the high-value invariants identified in
//! `docs/joule-db/cow-mvcc-design.md` §8:
//!
//! 1. **No torn reads under concurrency.** A writer commits in a
//!    tight loop while many reader threads each hold a `DbSnapshot`.
//!    Every snapshot read must produce a value consistent with the
//!    snapshot's pinned version — never a partial / mid-CoW view.
//!
//! 2. **Snapshot-version isolation.** Snapshots taken at different
//!    versions observe different generations of data. Two snapshots
//!    captured at v=N₁ and v=N₂ (N₂ > N₁) read independently; the
//!    writer's later commits cannot corrupt either.
//!
//! 3. **Crash recovery to last `Engine::sync`.** Writes performed
//!    after the last sync are not durable — a hard process drop
//!    followed by reopen must reflect only the state of the most
//!    recent committed `(committed_version, committed_root)`.

use joule_db_local::Database;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// 1 writer + 8 readers running for ~3 seconds of wall time. The
/// writer commits batches as fast as it can; each reader holds a
/// snapshot, performs many `get` calls against it, occasionally
/// refreshes, and asserts every observed value is consistent with
/// some past committed version.
///
/// "Consistent" here means: every key the snapshot can read must
/// be findable with the value some prior writer batch produced. We
/// encode the writer batch number in the value — the reader checks
/// the value parses to a non-negative integer and reads back the
/// same key→value mapping consistently across multiple gets.
#[test]
fn single_writer_many_readers_no_torn_reads() {
    let temp_dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(temp_dir.path()).unwrap());

    // Seed: every key starts at value "0".
    const NUM_KEYS: usize = 100;
    for i in 0..NUM_KEYS {
        db.put(format!("k{:03}", i).as_bytes(), b"0").unwrap();
    }
    db.sync().unwrap();

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(9)); // 1 writer + 8 readers
    let mut handles = Vec::new();

    // Writer thread: bumps every key's value to a monotonically
    // increasing batch number, syncs.
    {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            let mut batch: u64 = 1;
            while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                for i in 0..NUM_KEYS {
                    db.put(
                        format!("k{:03}", i).as_bytes(),
                        batch.to_string().as_bytes(),
                    )
                    .unwrap();
                }
                db.sync().unwrap();
                batch += 1;
            }
            batch
        }));
    }

    // 8 reader threads: each opens a snapshot, hammers reads against
    // it for ~3 seconds, occasionally refreshes. The invariant we
    // check inside each thread is per-snapshot value consistency:
    // every key the snapshot exposes must parse as an integer (no
    // garbage bytes), and successive reads of the same key from the
    // same snapshot must return the same value.
    for reader_id in 0..8 {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || -> u64 {
            barrier.wait();
            let mut reads: u64 = 0;
            let mut snap = db.open_snapshot().unwrap();
            let start = Instant::now();
            while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                // Sample a few keys, ensure consistent reads.
                for i in 0..NUM_KEYS {
                    let key = format!("k{:03}", i);
                    let v1 = snap
                        .get(key.as_bytes())
                        .expect("snapshot get must not error");
                    let v2 = snap
                        .get(key.as_bytes())
                        .expect("repeat snapshot get must not error");
                    assert_eq!(
                        v1, v2,
                        "reader {} saw torn read for key {} at snapshot v={}",
                        reader_id,
                        key,
                        snap.version()
                    );
                    let bytes = v1.expect("key must exist");
                    let s = std::str::from_utf8(&bytes)
                        .expect("value must be valid UTF-8");
                    let _: u64 = s
                        .parse()
                        .expect("value must parse as a u64 batch number");
                    reads += 1;
                }
                // Occasionally refresh — exercises the in-process
                // snapshot lifecycle (release + acquire + lockfile
                // churn).
                if reads.is_multiple_of(NUM_KEYS as u64 * 4) {
                    snap.refresh().expect("refresh should succeed");
                }
                if start.elapsed() > Duration::from_secs(3) {
                    break;
                }
            }
            reads
        }));
    }

    thread::sleep(Duration::from_secs(3));
    stop.store(true, std::sync::atomic::Ordering::SeqCst);

    // Collect and assert the writer made meaningful progress and
    // every reader completed many reads.
    let mut iter = handles.into_iter();
    let writer_batches = iter.next().unwrap().join().unwrap();
    assert!(
        writer_batches > 1,
        "writer should commit at least 1 batch, saw {}",
        writer_batches
    );
    for (reader_id, h) in iter.enumerate() {
        let reads = h.join().unwrap();
        assert!(
            reads > 0,
            "reader {} should complete at least 1 round, saw {}",
            reader_id,
            reads
        );
    }
}

/// **Phase 5 invariant: snapshot generations are independent.**
///
/// Take 5 snapshots, one per writer batch. Each captured at the
/// moment the writer has just committed batch N. After all 5 are
/// taken, the writer pushes 5 more batches. Then we assert: each
/// snapshot still observes its captured batch's data — not the
/// data from any later batch.
#[test]
fn snapshot_versions_observe_correct_data() {
    let temp_dir = TempDir::new().unwrap();
    let db = Database::open(temp_dir.path()).unwrap();

    const NUM_KEYS: usize = 20;
    let mut snapshots = Vec::new();

    // 5 generations: each batch updates every key's value to the
    // batch number's stringified form. Capture a snapshot after
    // every successful sync.
    for batch in 1..=5u64 {
        for i in 0..NUM_KEYS {
            db.put(
                format!("k{:02}", i).as_bytes(),
                batch.to_string().as_bytes(),
            )
            .unwrap();
        }
        db.sync().unwrap();
        let snap = db.open_snapshot().unwrap();
        assert_eq!(snap.version(), batch);
        snapshots.push((batch, snap));
    }

    // Push 5 more generations of writes. None of the existing
    // snapshots' captured roots should be affected.
    for batch in 6..=10u64 {
        for i in 0..NUM_KEYS {
            db.put(
                format!("k{:02}", i).as_bytes(),
                batch.to_string().as_bytes(),
            )
            .unwrap();
        }
        db.sync().unwrap();
    }

    // Verify each captured snapshot still reads its own generation.
    for (batch, snap) in &snapshots {
        for i in 0..NUM_KEYS {
            let key = format!("k{:02}", i);
            let val = snap
                .get(key.as_bytes())
                .unwrap()
                .expect("key must exist in snapshot");
            assert_eq!(
                val,
                batch.to_string().into_bytes(),
                "snapshot v={} key {} returned v={}'s data — \
                 should still see its own generation",
                batch,
                key,
                std::str::from_utf8(&val).unwrap_or("?")
            );
        }
    }

    // Drop snapshots in reverse order; engine's deferred frees can
    // now drain. Verify the engine remains usable (no double-free,
    // no panic) by performing one more committed write.
    drop(snapshots);
    for i in 0..NUM_KEYS {
        db.put(format!("k{:02}", i).as_bytes(), b"final").unwrap();
    }
    db.sync().unwrap();
    assert_eq!(
        db.get(b"k00").unwrap().as_deref(),
        Some(&b"final"[..]),
        "engine must remain healthy after all snapshots drop"
    );
}

/// **Phase 5 invariant: crash recovery == last `Engine::sync`.**
///
/// Two phases. Phase A: write some keys, sync. Drop the database.
/// Phase B: reopen, write more keys but **don't** sync. Drop again
/// without sync. Reopen — only Phase A's keys should be visible;
/// Phase B's writes were never committed and must not appear.
///
/// This validates that committed_version + committed_root in
/// meta.wdb v1 alone determine the visible state, not whatever
/// happens to be in the page cache or partial WAL records.
#[test]
fn crash_recovery_observes_only_committed_meta() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_path_buf();

    // Phase A: persist some keys via sync.
    {
        let db = Database::open(&path).unwrap();
        db.put(b"phase_a_key1", b"committed").unwrap();
        db.put(b"phase_a_key2", b"committed").unwrap();
        db.sync().unwrap();
        // Explicit drop ends Phase A.
    }

    // Phase B: write but don't sync. Drop simulates a crash.
    {
        let db = Database::open(&path).unwrap();
        // Sanity: phase A keys still readable on reopen.
        assert_eq!(
            db.get(b"phase_a_key1").unwrap().as_deref(),
            Some(&b"committed"[..])
        );
        assert_eq!(
            db.get(b"phase_a_key2").unwrap().as_deref(),
            Some(&b"committed"[..])
        );

        // Phase B writes — never sync.
        db.put(b"phase_b_key1", b"uncommitted").unwrap();
        db.put(b"phase_b_key2", b"uncommitted").unwrap();
        // No db.sync() here. Drop without committing.
    }

    // Final reopen: only phase A's data should be present. Phase B's
    // writes were lost because they never reached `committed_meta`.
    {
        let db = Database::open(&path).unwrap();
        assert_eq!(
            db.get(b"phase_a_key1").unwrap().as_deref(),
            Some(&b"committed"[..]),
            "phase A keys must survive reopen"
        );
        assert_eq!(
            db.get(b"phase_a_key2").unwrap().as_deref(),
            Some(&b"committed"[..]),
            "phase A keys must survive reopen"
        );
        // Phase B writes were never sync'd — they may or may not
        // appear depending on WAL behavior. The contract is that
        // only `committed_version`'s root determines the visible
        // state, so reads from that root must NOT include phase B
        // (phase B never made it into a CoW root chain that meta
        // points to).
        assert_eq!(
            db.get(b"phase_b_key1").unwrap(),
            None,
            "phase B writes must NOT survive a crash without sync"
        );
        assert_eq!(
            db.get(b"phase_b_key2").unwrap(),
            None,
            "phase B writes must NOT survive a crash without sync"
        );

        // The committed_version must reflect Phase A's sync only.
        let cm = db.engine().current_committed_meta();
        assert_eq!(
            cm.committed_version, 1,
            "only phase A's sync should be reflected in committed_version"
        );
    }
}

/// **CoW MVCC Phase 7.** Range scans against a snapshot must
/// observe only data committed at the snapshot's pinned version.
/// Same property as `Snapshot::get`, but exercises the iterator
/// path (`Snapshot::range` / `prefix_scan` / `scan`).
///
/// This is the read API scholar-server's handlers use for
/// listings, prefix lookups, and id-range queries.
#[test]
fn snapshot_range_scan_pinned_to_committed_root() {
    use joule_db_core::index::{Bound, ScanDirection};

    let temp_dir = TempDir::new().unwrap();
    let db = Database::open(temp_dir.path()).unwrap();

    // Sync 1: seed three keys under the same prefix.
    db.put(b"work:0001", b"v1-a").unwrap();
    db.put(b"work:0002", b"v1-b").unwrap();
    db.put(b"work:0003", b"v1-c").unwrap();
    db.sync().unwrap();

    let snap = db.open_snapshot().unwrap();
    let v1 = snap.version();

    // Full scan via the snapshot — should yield exactly the 3 seeds.
    let collected: Vec<_> = snap
        .scan(ScanDirection::Forward)
        .unwrap()
        .map(|r| {
            let e = r.unwrap();
            (e.key, e.value)
        })
        .collect();
    assert_eq!(collected.len(), 3);
    assert_eq!(collected[0].0, b"work:0001");
    assert_eq!(collected[0].1, b"v1-a");

    // Sync 2: writer adds + overwrites. Snapshot must NOT see them.
    db.put(b"work:0004", b"v2-d").unwrap();
    db.put(b"work:0001", b"v2-overwrite").unwrap();
    db.sync().unwrap();

    // Re-scan via the same snap — still pinned to v1.
    let still_seen: Vec<_> = snap
        .scan(ScanDirection::Forward)
        .unwrap()
        .map(|r| r.unwrap().value)
        .collect();
    assert_eq!(still_seen.len(), 3, "snap must not see v2 work:0004");
    assert_eq!(
        still_seen[0],
        b"v1-a",
        "snap must not see v2 overwrite of work:0001"
    );

    // Range scan: same isolation property.
    let range_seen: Vec<_> = snap
        .range(
            Bound::Included(b"work:0002".as_slice()),
            Bound::Included(b"work:0003".as_slice()),
            ScanDirection::Forward,
        )
        .unwrap()
        .map(|r| r.unwrap().value)
        .collect();
    assert_eq!(range_seen, vec![b"v1-b".to_vec(), b"v1-c".to_vec()]);

    // Prefix scan: same isolation property.
    let prefix_seen: Vec<_> = snap
        .prefix_scan(b"work:")
        .unwrap()
        .map(|r| r.unwrap().key)
        .collect();
    assert_eq!(prefix_seen.len(), 3);
    assert!(!prefix_seen.contains(&b"work:0004".to_vec()));

    assert_eq!(snap.version(), v1, "snap version must not have advanced");
}

/// **Merge-overflow regression test.** Previously, `MIN_KEYS=64`
/// (count-based) triggered aggressive merges even when leaves were
/// byte-size-limited to 8-16 keys. Merging two byte-fat half-leaves
/// produced a post-merge node larger than the 64KB page — the
/// storage encode step then panicked with `PageSizeExceeded`.
///
/// This test pre-populates 200 keys × 2KB values (about 400KB of
/// payload spread across many leaves), then deletes most of them.
/// The new byte-size-based rebalance logic skips merges that would
/// overflow the page; the test must complete without errors and
/// every surviving key must be readable.
///
/// Reproduces the failure scenario from the original Phase 4.5 test
/// before that test was narrowed; verifies the fix.
#[test]
fn merge_does_not_overflow_page_under_byte_fat_values() {
    let temp_dir = TempDir::new().unwrap();
    let db = Database::open(temp_dir.path()).unwrap();

    // Each value is 2KB — well below max_inline_value_size (~16KB)
    // so values stay inline in leaves. With ~16 keys per leaf
    // before split, leaves are byte-size-limited.
    let big_val = vec![b'X'; 2048];

    for i in 0..200u32 {
        db.put(format!("k{:05}", i).as_bytes(), &big_val).unwrap();
    }
    db.sync().unwrap();

    // Sanity: data round-trips before any deletes.
    for i in 0..200u32 {
        assert!(
            db.get(format!("k{:05}", i).as_bytes())
                .unwrap()
                .is_some(),
            "seed key k{:05} must exist",
            i
        );
    }

    // Delete 190 of the 200 keys. Pre-fix this triggered cascading
    // merges that overflowed pages and panicked at encode time.
    for i in 0..190u32 {
        db.delete(format!("k{:05}", i).as_bytes()).unwrap();
    }
    db.sync().unwrap();

    // Surviving keys (190..200) must still be readable.
    for i in 190..200u32 {
        let val = db
            .get(format!("k{:05}", i).as_bytes())
            .unwrap()
            .expect("surviving key must exist");
        assert_eq!(
            val, big_val,
            "surviving key k{:05} value corrupted",
            i
        );
    }

    // Deleted keys must be gone.
    for i in 0..190u32 {
        assert_eq!(
            db.get(format!("k{:05}", i).as_bytes()).unwrap(),
            None,
            "deleted key k{:05} should not exist",
            i
        );
    }

    // The committed_version should reflect both syncs.
    let cm = db.engine().current_committed_meta();
    assert!(
        cm.committed_version >= 2,
        "expected at least 2 syncs, saw committed_version={}",
        cm.committed_version
    );
}

/// **Production bug regression (2026-04-28).** Previously
/// `FileBackend::read_committed_meta` returned the in-memory mirror,
/// not a disk re-read. That broke the cross-process refresh
/// semantic: scholar-server (process A, reader) never observed
/// scholar-ingestd's (process B, writer) commits because process A's
/// in-memory mirror was set at open time and never updated by peer
/// processes. The refresher silently no-op'd.
///
/// This test simulates the cross-process pattern by **opening two
/// independent `Database` instances against the same path**, both in
/// the same process. They have separate FileBackend in-memory state
/// (separate buffer pools, separate committed_meta mirrors), so
/// changes made through one are only visible to the other through
/// the on-disk meta.wdb — exactly the cross-process scenario.
#[test]
fn refresh_picks_up_peer_writes_via_disk_not_in_memory_mirror() {
    let temp_dir = TempDir::new().unwrap();

    // Reader process: opens DB, opens snapshot at version 0.
    let reader_db = Database::open(temp_dir.path()).unwrap();
    let mut reader_snap = reader_db.open_snapshot().unwrap();
    let initial = reader_snap.version();
    assert_eq!(reader_snap.get(b"k1").unwrap(), None);

    // Writer process: completely separate Database opening the same
    // files. Performs a put + sync. The writer's FileBackend writes
    // v1 to meta.wdb on disk; the reader's FileBackend has no
    // in-memory knowledge of this.
    {
        let writer_db = Database::open(temp_dir.path()).unwrap();
        writer_db.put(b"k1", b"peer-write").unwrap();
        writer_db.sync().unwrap();
        // writer drops here, releasing its resources.
    }

    // Without the bug fix, reader_snap.refresh() would read the
    // reader's in-memory mirror (still empty) and silently leave
    // the snapshot pinned at version 0. With the fix, refresh reads
    // meta.wdb from disk and advances.
    reader_snap.refresh().unwrap();
    assert!(
        reader_snap.version() > initial,
        "refresh must advance past initial version after peer commit \
         (saw {} -> {}); regression of the production bug fixed 2026-04-28",
        initial,
        reader_snap.version()
    );
    assert_eq!(
        reader_snap.get(b"k1").unwrap().as_deref(),
        Some(&b"peer-write"[..]),
        "post-refresh, peer's writes must be visible to reader"
    );
}

/// **CoW MVCC Phase 6.** A snapshot opened *before* the writer's
/// commit must NOT see the new data — but a `refresh()` call
/// against the same snapshot *must* pick up the latest committed
/// state via the backend's atomic `meta.wdb` re-read.
///
/// In a real deployment this is the cross-process semantic:
/// scholar-server's snapshot is opened against an engine in process
/// A; scholar-ingestd commits to the same database from process B;
/// scholar-server's periodic `refresh()` picks up B's commits via
/// the disk meta record. Here we exercise it in-process — the
/// disk path is the same.
#[test]
fn snapshot_refresh_picks_up_writer_commits_via_backend() {
    let temp_dir = TempDir::new().unwrap();
    let db = Database::open(temp_dir.path()).unwrap();

    // Commit 1: seed.
    db.put(b"k1", b"v1-initial").unwrap();
    db.sync().unwrap();

    let mut snap = db.open_snapshot().unwrap();
    let v1 = snap.version();
    assert!(v1 >= 1);
    assert_eq!(
        snap.get(b"k1").unwrap().as_deref(),
        Some(&b"v1-initial"[..])
    );

    // Commit 2: writer adds a new key.
    db.put(b"k2", b"v2-new").unwrap();
    db.sync().unwrap();

    // Pre-refresh: snapshot still pinned at v1.
    assert_eq!(snap.version(), v1);
    assert_eq!(
        snap.get(b"k2").unwrap(),
        None,
        "snapshot must NOT see v2 commit before refresh"
    );

    // Refresh — disk meta.wdb gets re-read; snap advances.
    snap.refresh().unwrap();
    assert!(
        snap.version() > v1,
        "refresh should advance to a later committed_version (saw {} -> {})",
        v1,
        snap.version()
    );
    assert_eq!(
        snap.get(b"k2").unwrap().as_deref(),
        Some(&b"v2-new"[..]),
        "post-refresh snapshot should see the new commit"
    );
    // Phase A's data still readable post-refresh.
    assert_eq!(
        snap.get(b"k1").unwrap().as_deref(),
        Some(&b"v1-initial"[..])
    );
}

/// **Phase 5 invariant: snapshot under writer churn keeps consistent
/// view of a multi-key transaction.**
///
/// Two-key invariant: A and B always sum to 1000. The writer
/// performs transfers: A -= n; B += n; sync. A reader holding a
/// snapshot at any single committed version must always see
/// A + B == 1000 — never a half-applied transfer.
#[test]
fn snapshot_preserves_multi_key_invariant_under_writer_churn() {
    let temp_dir = TempDir::new().unwrap();
    let db = Arc::new(Database::open(temp_dir.path()).unwrap());

    // Seed: A=600, B=400. Sum is 1000.
    db.put(b"acct_a", b"600").unwrap();
    db.put(b"acct_b", b"400").unwrap();
    db.sync().unwrap();

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(2));
    let writer = {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            let mut a: i64 = 600;
            let mut b: i64 = 400;
            let mut transfers: u64 = 0;
            while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                let amount = (transfers % 50) as i64 + 1;
                if a >= amount {
                    a -= amount;
                    b += amount;
                } else {
                    b -= amount;
                    a += amount;
                }
                db.put(b"acct_a", a.to_string().as_bytes()).unwrap();
                db.put(b"acct_b", b.to_string().as_bytes()).unwrap();
                db.sync().unwrap();
                transfers += 1;
            }
            transfers
        })
    };

    let reader = {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || -> u64 {
            barrier.wait();
            let mut snap = db.open_snapshot().unwrap();
            let mut checks: u64 = 0;
            let start = Instant::now();
            while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                let a_bytes = snap
                    .get(b"acct_a")
                    .unwrap()
                    .expect("acct_a must exist in snapshot");
                let b_bytes = snap
                    .get(b"acct_b")
                    .unwrap()
                    .expect("acct_b must exist in snapshot");
                let a: i64 = std::str::from_utf8(&a_bytes)
                    .unwrap()
                    .parse()
                    .unwrap();
                let b: i64 = std::str::from_utf8(&b_bytes)
                    .unwrap()
                    .parse()
                    .unwrap();
                assert_eq!(
                    a + b,
                    1000,
                    "snapshot at v={} saw torn transfer: a={} b={} sum={}",
                    snap.version(),
                    a,
                    b,
                    a + b,
                );
                if checks.is_multiple_of(64) {
                    snap.refresh().expect("refresh should succeed");
                }
                checks += 1;
                if start.elapsed() > Duration::from_secs(2) {
                    break;
                }
            }
            checks
        })
    };

    thread::sleep(Duration::from_secs(2));
    stop.store(true, std::sync::atomic::Ordering::SeqCst);

    let transfers = writer.join().unwrap();
    let checks = reader.join().unwrap();
    assert!(
        transfers > 1,
        "writer should perform > 1 transfer in 2s, saw {}",
        transfers
    );
    assert!(
        checks > 0,
        "reader should perform > 0 invariant checks, saw {}",
        checks
    );
}
