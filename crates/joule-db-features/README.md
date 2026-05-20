# joule-db-features

Specialized feature modules for JouleDB.

`joule-db-features` is the umbrella for cross-cutting feature surfaces that don't fit cleanly into the core, query, or HDC crates: vector indexes, graph operators, time-series operators, full-text search, columnar layouts, and SIMD primitives.

## Module map

| Module | Role |
|---|---|
| [`vector.rs`](src/vector.rs) | Vector indexes — HNSW, IVF, flat |
| [`graph.rs`](src/graph.rs), [`graph/`](src/graph/) | Graph operators — traversal, shortest-path, centrality |
| [`timeseries/`](src/timeseries/) | Time-series operators — downsampling, gap fill, windowed aggregates |
| [`fulltext.rs`](src/fulltext.rs) | Full-text search — tokenization, tf-idf, BM25 |
| [`embeddings.rs`](src/embeddings.rs) | Embedding storage + retrieval |
| [`columnar.rs`](src/columnar.rs) | Columnar layout helpers |
| [`simd.rs`](src/simd.rs) | SIMD-accelerated primitives |
| [`persistence.rs`](src/persistence.rs) | Feature persistence helpers |

## Tests

119 tests in `src/`.

## Server bridge

[`joule-db-server::features_bridge`](../joule-db-server/src/features_bridge.rs) wires these features into the wire protocol.

## See also

- [joule-db-core](../joule-db-core/)
- [joule-db-domains](../joule-db-domains/) — domain-specific encodings layered on top of these generic features
- [joule-db-hdc](../joule-db-hdc/) — HDC substrate for vector / similarity ops
