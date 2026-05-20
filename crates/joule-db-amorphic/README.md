# joule-db-amorphic

Amorphic Database — just-in-time structure materialization via HDC.

`joule-db-amorphic` is the schema-fluid layer of JouleDB. Instead of demanding a fixed schema at write time, it stores data as hyperdimensional vectors via [`joule-db-hdc`](../joule-db-hdc/) and materializes concrete structs on demand at query time. The result is a single substrate that simultaneously serves relational, document, graph, vector, time-series, and full-text workloads — without paying the join cost of a polyglot stack.

This is the largest of the JouleDB specialized crates: **~51K LOC, 684 tests, 92 source files**.

## Module map

### Core
| Module | Role |
|---|---|
| [`columnar.rs`](src/columnar.rs) | Columnar materialization path |
| [`materialized.rs`](src/materialized.rs) | Materialized-view machinery |
| [`memory.rs`](src/memory.rs) | In-memory amorphic store |
| [`durable.rs`](src/durable.rs) | Disk-backed amorphic store |
| [`persistence.rs`](src/persistence.rs) | Persistence layer |
| [`partition.rs`](src/partition.rs) | Partition routing |
| [`distributed.rs`](src/distributed.rs) | Distributed-store mode |
| [`distribution.rs`](src/distribution.rs) | Distribution / sharding strategy |
| [`platform.rs`](src/platform.rs) | Platform abstraction |
| [`optimizer.rs`](src/optimizer.rs) | Amorphic-query optimizer |
| [`batch.rs`](src/batch.rs) | Batched writes |
| [`tx.rs`](src/tx.rs) | Transactional surface |
| [`gpu.rs`](src/gpu.rs) | GPU compute path |
| [`hybrid_search.rs`](src/hybrid_search.rs) | Hybrid vector + keyword search |
| [`hologram_sync.rs`](src/hologram_sync.rs) | Holographic memory synchronization |
| [`auth.rs`](src/auth.rs) | Auth layer |
| [`content_id.rs`](src/content_id.rs) | Content-addressable IDs |

### Vertical adapters
| Module | Vertical |
|---|---|
| [`ai/`](src/ai/) | AI workloads |
| [`accessibility.rs`](src/accessibility.rs) | A11y metadata |
| [`ad_targeting.rs`](src/ad_targeting.rs) | Ad-targeting workloads |
| [`cdn/`](src/cdn/) | CDN integration |
| [`events.rs`](src/events.rs) | Event store |
| [`knowledge/`](src/knowledge/) | Knowledge-graph workloads |
| [`moderation.rs`](src/moderation.rs) | Content moderation |
| [`royalty.rs`](src/royalty.rs) | Royalty/payments tracking |

## Feature flags

| Feature | Purpose |
|---|---|
| `durable` | Enable disk-backed durable store |
| `gpu` | GPU compute path |
| `distributed` | Distributed-store mode |

## Benchmarks

10 benches in [`benches/`](benches/) including TPCH, LDBC, YCSB, ANN, multi-model, content-infra, competitive, and stress harnesses.

## Tests

684 tests in `src/`, plus [`tests/shard_routing.rs`](tests/shard_routing.rs) for routing correctness.

## See also

- [joule-db-core](../joule-db-core/)
- [joule-db-hdc](../joule-db-hdc/) — the HDC substrate amorphic encodes against
- [joule-db-query](../joule-db-query/) — `amorphic_executor.rs` ties this crate into the query engine
