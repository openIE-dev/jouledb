# JouleDB Recovery Runbook

**Version 1.1 — 2026-05-19**
**Scope:** how to recover a JouleDB instance that won't open or returns corrupt results
**Sister docs:** [`CLI-REFERENCE.md`](CLI-REFERENCE.md), [`crates/joule-db-local/src/bin/joule_db_recover.rs`](../../crates/joule-db-local/src/bin/joule_db_recover.rs), `the-fix-1.md`, `the-fix-2.md`

This runbook is what you reach for when a JouleDB instance is broken. It assumes you've already verified the underlying disk is healthy and the WAL replay isn't enough.

> **v1.1 — index auto-recovery landed (§10.1).** After a JouleDB forensic recovery salvages a *sparse subset* of rows, the search index (tantivy) still has postings for rows that no longer exist. As of 2026-05-19 `scholar-server` **auto-detects and self-heals** this — see §11 below. The operator no longer has to run `rebuild-index` by hand for the post-recovery sparse-fill case.

---

## 1. Decision tree

```
JouleDB won't open or returns corrupt results
  │
  ▼
Does the WAL replay help?     ──── yes → done (engine handles it automatically)
  │
  no
  ▼
Is meta.wdb corrupted/wrong?  ──── yes → §2  joule-db-recover --write-recovery-meta
  │
  no
  ▼
Is the internal-node tree     ──── yes → §3  joule-db-recover --write-synthetic-recovery
corrupted but leaves intact?         OR §4  joule-db-recover --rebuild-tree
  │
  no
  ▼
Is the catalog itself broken? ──── yes → §5  scholar-recover rebuild
  │
  no
  ▼
Data file corrupted?           ──── §6  restore from backup
```

---

## 2. Mode 1: `meta.wdb` rewrite (least invasive)

**Symptom:** `meta.wdb` points at a small / wrong tree but the original tree's pages still exist in `data.wdb`.

```bash
# Step 1: scan, read-only — safe to run against a live database
joule-db-recover /path/to/db
# Output: page classification, candidate roots, "biggest reachable subtree: page 1234567"

# Step 2: rewrite meta to point at the biggest candidate
joule-db-recover /path/to/db --write-recovery-meta

# Step 3: optionally pick a specific root
joule-db-recover /path/to/db --write-recovery-meta --root 9107416
```

**Safety:** the previous `meta.wdb` is preserved as `meta.wdb.pre-recover-<unix_ts>`. The new file is written via tmp + fsync + rename.

**Validate:** restart the JouleDB server, run a few `COUNT(*)` queries against expected tables, compare row counts to your last known-good metrics.

---

## 3. Mode 2: synthetic recovery (multiple table roots)

**Symptom:** multiple tables had separate orphaned subtrees; no single existing root reaches all of them.

```bash
joule-db-recover /path/to/db --write-synthetic-recovery
```

What it does: allocates a fresh page id, writes a synthetic B-tree internal page that points at the largest reachable subtree **per table prefix**, then rewrites `meta.wdb` to point at the synthetic page.

This is what the Scholar 2026-05-02 incident used. The corrupt root had stranded `scholar_works` (4M rows), `scholar_concepts`, `scholar_citations`, and `scholar_authority` in separate orphan subtrees; synthetic recovery rebuilt one consistent root over all of them.

**Validate:**
- `meta.wdb.pre-recover-<ts>` exists
- `SELECT count(*) FROM scholar_works` returns a number close to your last known total
- spot-check a few queries against known IDs

---

## 4. Mode 3: full rebuild (most invasive non-data-loss option)

**Symptom:** internal-node tree fully corrupted; even synthetic recovery can't find consistent roots; leaves are intact.

```bash
joule-db-recover /path/to/db --rebuild-tree
```

What it does:
1. Sorts ALL leaf pages by first key
2. Deduplicates overlapping ranges (skip-on-overlap, recent-wins tie-break)
3. Bulk-builds a fresh internal-node tree on top of the consistent leaf set
4. Allocates fresh page ids for all new internals (~`L/200` pages where `L` is the surviving leaf count)
5. Points `meta.wdb` at the new root

Validate carefully — this rebuild has the highest risk of data shape changes. Compare row counts, run application smoke tests, verify indexes get rebuilt on next server start.

---

## 5. Mode 4: `scholar-recover rebuild` (catalog-broken, last resort short of restore)

**Symptom:** the catalog B-tree references dangling pages; WAL is empty; even `joule-db-recover --rebuild-tree` can't proceed because the engine can't even open the database in inspect-only mode.

```bash
scholar-recover rebuild --src /broken/data.wdb --dest /fresh/db/dir
```

What it does: bypasses the B-tree entirely. Treats `data.wdb` as an opaque sequence of 64KB pages. For each `BTreeLeaf` page, deserializes the body and emits `(key, value)` pairs whose keys start with `row::{table}\0` into a fresh JouleDB.

Indexes are dropped (they'll be recreated on the fresh DB's next boot from row data). Missing leaf pages → those specific rows lost; the rest survive.

Cost: hours for a multi-GB database (writes via the standard clean B-tree path, so write-amplified). Run inside `tmux` / `nohup`.

---

## 6. Mode 5: restore from backup

**Symptom:** `data.wdb` itself is corrupted at the page level (magic bytes failing, large pages full of zeros, etc.). No tool can recover from a corrupt data file directly.

```bash
# Stop the server
systemctl stop jouledb
# OR: kubectl scale statefulset my-cluster -n jouledb --replicas=0

# Restore from the most recent backup
cp /backups/jouledb-2026-05-17-0200.wdb /var/lib/jouledb/data.wdb
cp /backups/jouledb-2026-05-17-0200.meta.wdb /var/lib/jouledb/meta.wdb

# Optional: replay WAL forward to recover writes since the backup
joule db wal replay /var/lib/jouledb/wal.wdb --from 2026-05-17T02:00:00 --to 2026-05-17T08:30:00

# Restart
systemctl start jouledb
```

In-tree backup snapshots live in [`crates/joule-db-server/backups/`](../../crates/joule-db-server/backups/) for testing; production backup location is per your deployment. See [`RUNBOOK-BACKUP.md`](RUNBOOK-BACKUP.md).

---

## 7. In a Kubernetes pod

If the JouleDB pod is in `CrashLoopBackOff` because the engine refuses to open:

```bash
# Bypass the entrypoint and exec the recovery tool directly
kubectl exec -it my-cluster-0 -n jouledb -- joule-db-recover scan /data/joule.db

# If the pod won't start at all, override the entrypoint
kubectl debug -it my-cluster-0 -n jouledb \
    --image=ghcr.io/openie/jouledb:latest \
    --target=my-cluster-0 \
    -- /bin/sh

# Inside the debug pod
joule-db-recover scan /data/joule.db
joule-db-recover /data/joule.db --write-recovery-meta
```

For full cluster lifecycle in K8s, see [`CLOUD-OPERATOR.md`](CLOUD-OPERATOR.md).

---

## 8. Post-recovery checklist

- [ ] Server starts cleanly and accepts connections
- [ ] `SELECT count(*) FROM <each_known_table>` returns sane numbers
- [ ] Application smoke tests pass
- [ ] Tantivy indexes rebuild (the server does this lazily on first query that needs them)
- [ ] Energy receipts are flowing again — check the ledger
- [ ] `meta.wdb.pre-recover-<ts>` is preserved on disk (don't delete until you've validated)
- [ ] If you recovered via Mode 5 (backup): note the WAL replay window — anything after the WAL truncation point is lost

---

## 9. When to call for help

If after Mode 5 you still can't recover what you need, the data is genuinely lost from this database. Check:

- Other JouleDB replicas (if running Raft)
- Source-of-record systems (especially relevant for Scholar — the publishing graph is rebuildable from CC0/OA upstream sources; see [`project_scholar_reingest_repair_2026_05_03`](../../../.claude/projects/-Users-dcharlot-data-share-vibe-coding-joulesperbit/memory/project_scholar_reingest_repair_2026_05_03.md))
- Recent backups (you have a backup, right?)

---

## 10. Search-index auto-recovery (§10.1 — landed 2026-05-19)

After any of recovery modes 1–5 above, the JouleDB row set is smaller than it was. The tantivy search index, built before the incident, still has postings pointing at rows that no longer exist — the **sparse-fill** state. Previously the operator had to run `scholar-ingestd rebuild-index` and restart `scholar-server` by hand. That step is now automatic.

### How it works

`scholar-server` runs a background **index health monitor** ([`crates/scholar-server/src/main.rs`](../../crates/scholar-server/src/main.rs), `spawn_index_health_monitor`):

1. It reuses the existing row-count cache (the `spawn_row_count_refresher` task) — **no extra scan, zero added I/O on the hot path**.
2. Every `SCHOLAR_INDEX_HEALTH_REFRESH_S` (default 120 s) it compares the live tantivy doc count against the cached `scholar_works` row count.
3. The healthy invariant is `tantivy_docs <= jouledb_rows` (the rebuild skips empty-content rows, so the index can only ever have *fewer* docs than rows). When `tantivy_docs > jouledb_rows × (1 + tolerance)` (tolerance default 0.05), that can only mean rows vanished out from under the index → **sparse-fill detected**.
4. The index transitions `Healthy → Degraded`. It **keeps serving** (a stale hit beats a hard 503) but every search response carries a hint: *"index is recovering from a database recovery…"*.
5. A **throttled background rebuild** kicks off: a dedicated blocking thread, reduced writer heap (`SCHOLAR_INDEX_REBUILD_HEAP_MB`, default 64 MB so it can't draw a jetsam kill mid-rebuild), building into a sibling `<index_dir>.rebuilding` directory.
6. On success the directory is swapped via the Unix rename dance (rotate old aside → move new in → open fresh reader → hot-swap the `Arc` → drop old reader; its file descriptors stay valid until in-flight queries finish). Health returns to `Healthy`. **No process restart.**
7. On failure it falls back to `Degraded` (or `Unavailable` if there was never a handle) and the monitor retries next cycle.

### State machine

```
Healthy ──(tantivy_docs > rows×1.05)──► Degraded ──► Rebuilding ──(swap ok)──► Healthy
   ▲                                                      │
   └──────────────(monitor sees rows caught up)───────────┘ (on failure: back to Degraded, retry)

Unavailable ──(DB has rows, no index)──► Rebuilding ──► Healthy
```

### Operator knobs

| Env var | Default | Effect |
|---|---|---|
| `SCHOLAR_INDEX_HEALTH_REFRESH_S` | 120 | Monitor poll interval (seconds) |
| `SCHOLAR_INDEX_STALE_TOLERANCE` | 0.05 | Fractional slack above row count before triggering |
| `SCHOLAR_INDEX_REBUILD_HEAP_MB` | 64 | Tantivy writer heap during the throttled rebuild |
| `SCHOLAR_DISABLE_INDEX_AUTORECOVERY` | unset | Set to `1` to disable auto-recovery entirely (manual `rebuild-index` only) |
| `SCHOLAR_SKIP_ROW_COUNTS` | unset | Also disables auto-recovery (it depends on the row-count cache) |

### When you still rebuild by hand

- You set `SCHOLAR_DISABLE_INDEX_AUTORECOVERY=1` (or `SCHOLAR_SKIP_ROW_COUNTS=1` on a very large DB).
- You want the index rebuilt *immediately* rather than within one monitor cycle + rebuild duration — run `scholar-ingestd rebuild-index`.
- The embed index (`scholar-embed`) is also stale — auto-recovery covers the **BM25** index only; `rebuild-embed` is still manual (the embedding model is the expensive part; auto-triggering it is intentionally out of scope).

### Verifying

```bash
# Health is exposed on the metrics surface
curl -s localhost:8080/metrics | grep scholar_index
# Logs narrate the transition
journalctl -u scholar-server | grep -E "sparse-fill|auto-recovery"
```

---

## 11. See also

- [`CLI-REFERENCE.md`](CLI-REFERENCE.md) — full CLI surface
- [`RUNBOOK-BACKUP.md`](RUNBOOK-BACKUP.md) — backup procedures
- [`RUNBOOK-CLUSTERING.md`](RUNBOOK-CLUSTERING.md) — multi-replica deployment
- [`crates/joule-db-local/src/bin/joule_db_recover.rs`](../../crates/joule-db-local/src/bin/joule_db_recover.rs) — source of truth for the binary
- [`crates/scholar-server/src/main.rs`](../../crates/scholar-server/src/main.rs) — `spawn_index_health_monitor`, `spawn_throttled_rebuild`, `is_sparse_fill`
- [`the-fix-1.md`](../../the-fix-1.md), [`the-fix-2.md`](../../the-fix-2.md) — Scholar incident narratives
