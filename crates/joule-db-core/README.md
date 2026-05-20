# joule-db-core

Platform-agnostic core database engine for JouleDB.

`joule-db-core` is the foundation of the JouleDB ecosystem — a B-tree storage engine with ACID transactions, MVCC copy-on-write commits, and trait-based abstractions for storage, indexes, persistence, and compute. It has zero platform dependencies (no `tokio`, no `wasm_bindgen`, no OS-specific code), so the same engine compiles for native, browser (via [`joule-db-browser`](../joule-db-browser/)), edge (via [`joule-db-edge`](../joule-db-edge/)), and MCU targets.

## Design principles

1. **Zero platform dependencies** — pluggable backends via traits
2. **Correctness first** — ACID guarantees over raw performance
3. **Cross-platform** — native, WASM, embedded all from one source
4. **GPU-native** — compute abstractions designed for accelerators from day one
5. **Energy-aware** — every operation accountable in joules ([`joule-db-energy`](../joule-db-energy/))

## What's inside

| Module | Role |
|---|---|
| `engine/` | B-tree implementation: depth-bounded descent, MVCC CoW, Phase 5/6 atomic publish |
| `storage/` | Page format, buffer pool (16-shard LRU), overflow chains, mmap extents |
| `tx/` | Transactions: `BeginTx`, `CommitTx`, `RollbackTx`, snapshot registry |
| `persistence/` | WAL, snapshots, recovery, per-sync truncate |
| `index/` | Pluggable indexes (B-tree, hash, vector, full-text via [`joule-db-features`](../joule-db-features/)) |
| `query/` | Query primitives shared by [`joule-db-query`](../joule-db-query/) |
| `concurrency/` | Lock-free primitives, latch-free traversal helpers |
| `resilience/` | Cycle protection, depth bounds, orphan-page recovery |
| `snapshot.rs` | MVCC snapshot construction + cross-process registry |
| `temporal/` | Time-travel reads (system-versioned predicate surface) |
| `catalog/` | Schema catalog, table metadata |
| `allocator/` | Page allocation, free-list management |
| `encryption.rs` | At-rest encryption hooks (key supplied by backend) |
| `error.rs` | `EngineError`, `Result<T>` |
| `types/` | Page IDs, transaction IDs, version numbers |

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                      JouleDB Ecosystem                          │
├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
│   Browser   │    MCU      │   Desktop   │   Server    │  Edge  │
│   (WASM)    │ (ARM/RISC)  │ (Win/Mac/Ln)│  (Linux)    │  (IoT) │
├─────────────┴─────────────┴─────────────┴─────────────┴────────┤
│                    Unified Query Layer                          │
├─────────────────────────────────────────────────────────────────┤
│              GPU/NPU/TPU Acceleration Layer                     │
│         (WebGPU | CUDA | Metal | Vulkan | SIMD)                 │
├─────────────────────────────────────────────────────────────────┤
│                   joule-db-core (THIS CRATE)                    │
│              (WAL | Snapshots | Recovery | B-tree)              │
└─────────────────────────────────────────────────────────────────┘
```

## Feature flags

| Feature | Default | Purpose |
|---|---|---|
| `async` | off | Async-trait support for storage backends |
| `async-tokio` | off | `tokio`-backed retry/timeout utilities |
| `serde` | off | `Serialize`/`Deserialize` on core types |
| `group-commit` | off | Group-commit batching for WAL fsync amortization |

## Usage

```rust,ignore
use joule_db_core::{Database, MemoryBackend};

// Open an in-memory database (real deployments use joule-db-local for disk)
let backend = MemoryBackend::new();
let db = Database::open(backend)?;

db.put(b"key", b"value")?;
let value = db.get(b"key")?;

let tx = db.begin()?;
tx.put(b"key1", b"value1")?;
tx.put(b"key2", b"value2")?;
tx.commit()?;
```

For disk-backed deployment, see [`joule-db-local`](../joule-db-local/). For the query layer (SQL, Cypher, CQL, GraphQL, Datalog, SPARQL, Gremlin), see [`joule-db-query`](../joule-db-query/). For the energy-aware AI cascade on top, see [`jouledb-ai-runtime`](../jouledb-ai-runtime/).

## Tests

350 `#[test]` / `#[tokio::test]` annotations in `src/`. Fuzz targets live in [`fuzz/`](./fuzz/). Cross-architecture byte-level determinism is verified by [`joule-compose-cross-arch.yml`](../../.github/workflows/joule-compose-cross-arch.yml).

## Recent hardening (May 2026)

- `abort_uncommitted()` frees orphan pages allocated by failed writes
- Point-lookup descents are iterative + depth-bounded (cycle-safe under corrupt parent pointers)
- CoW rollback on write failure unwinds page versions to pre-transaction state
- `prefix_count` tree-walk: row-count refresh ~1800× faster than the prior approach
- B-tree iterators terminate gracefully on `get_node` / `descend` errors instead of looping
- WAL truncate per-sync, not per-commit
- `JOULEDB_TOLERATE_CORRUPT_PAGES` escape hatch removed — the engine refuses corrupt pages upfront

## Forensic recovery

If `meta.wdb` is corrupted but `data.wdb` is intact, the [`joule-db-recover`](../joule-db-local/src/bin/joule_db_recover.rs) tool walks every page, classifies them, and rebuilds the root pointer. See `the-fix-1.md` / `the-fix-2.md` for the Scholar incident that drove its design.

## See also

- [WHITEPAPER-JOULEDB-2026-04.md](../../docs/WHITEPAPER-JOULEDB-2026-04.md) — v0.1 architectural whitepaper
- [MGAI-SPEC-DOMAIN-JOULEDB.md](../../docs/MGAI-SPEC-DOMAIN-JOULEDB.md) — domain audit pass
- [docs/joule-db/cow-mvcc-design.md](../../docs/joule-db/cow-mvcc-design.md) — Phase 5/6 atomic-publish design
