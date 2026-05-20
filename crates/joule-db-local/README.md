# joule-db-local

Native / local storage backend for JouleDB with file support.

`joule-db-local` is the disk-backed deployment of [`joule-db-core`](../joule-db-core/) — mmap-backed extent storage, WAL persistence, snapshot/recovery, and the forensic page-walker that recovered Scholar's 4M-row B-tree in May.

## Module map

| Module | Role |
|---|---|
| [`storage/`](src/storage/) | Mmap extent backend, file layout, page I/O |
| [`lsm/`](src/lsm/) | LSM-tree variant (write-heavy workloads) |
| [`server/`](src/server/) | Local-server harness for embedded deployments |
| [`bin/`](src/bin/) | Binaries — see below |
| [`bloom.rs`](src/bloom.rs) | Bloom filters for negative-lookup short-circuiting |
| [`recovery.rs`](src/recovery.rs) | WAL replay + recovery flow |
| [`snapshot_registry.rs`](src/snapshot_registry.rs) | Cross-process snapshot coordination |

## Binaries

- **`joule-db-recover`** ([src/bin/joule_db_recover.rs](src/bin/joule_db_recover.rs)) — forensic recovery tool. Walks every page in `data.wdb`, classifies them (`BTreeInternal`, `BTreeLeaf`, `Overflow`, `Free`), builds the parent→children adjacency graph, finds candidate roots (pages not referenced as any child), and BFS-measures the subtree size of each candidate. Use it when `meta.wdb` is corrupted but the data file is intact. `--write-recovery-meta` atomically rewrites `meta.wdb` to point at the largest candidate root. Scholar's May 2 recovery was driven by this binary.

## Recent hardening (May 2026)

- WAL truncate deferred from per-commit to per-sync (`57400f00c`)
- `JOULEDB_TOLERATE_CORRUPT_PAGES` escape hatch removed (`661a50c7a`)

## Tests

117 tests in `src/`, 1 integration test file in [`tests/`](./tests/).

## See also

- [joule-db-core](../joule-db-core/) — the engine being persisted
- `docs/jouledb/RUNBOOK-RECOVERY.md` *(in progress)*
- `the-fix-1.md`, `the-fix-2.md` (workspace root) — recovery narratives
