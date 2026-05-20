# joule-db-ledger

Blockchain-anchored energy receipt layer for JouleDB.

`joule-db-ledger` is the audit-trail layer — every joule the engine reports goes into an append-only Merkle-tree ledger, periodically batched and anchored to an external chain (Ethereum or any compatible backend). The result: cryptographically-verifiable energy receipts that downstream parties can audit without trusting the JouleDB operator.

## Module map

| Module | Role |
|---|---|
| [`receipt.rs`](src/receipt.rs) | `EnergyReceipt` type — what a single metered op records |
| [`merkle.rs`](src/merkle.rs) | Merkle-tree construction over receipt batches |
| [`collector.rs`](src/collector.rs) | Receipt collector — gathers raw joule events |
| [`batch.rs`](src/batch.rs) | Batching strategy (size / time / energy threshold) |
| [`committer.rs`](src/committer.rs) | Commits batches to the configured backend |
| [`backend.rs`](src/backend.rs) | Backend trait |
| [`backend_memory.rs`](src/backend_memory.rs) | In-memory backend (testing) |
| [`backend_file.rs`](src/backend_file.rs) | File-backed backend (single-machine durability) |
| [`backend_eth.rs`](src/backend_eth.rs) | Ethereum backend (public anchoring) |
| [`carbon.rs`](src/carbon.rs) | Carbon-accounting overlay (joules → kgCO₂e by region/time) |
| [`http.rs`](src/http.rs) | HTTP surface for ledger queries |
| [`error.rs`](src/error.rs) | Error types |

## Tests

70 tests in `src/`, plus [`tests/stress_ledger.rs`](tests/stress_ledger.rs) for stress.

## Server bridge

[`joule-db-server::ledger_bridge`](../joule-db-server/src/ledger_bridge.rs) wires the ledger into the wire protocol.

## See also

- [joule-db-energy](../joule-db-energy/) — joule measurement primitives that feed the ledger
- [jouledb-ai-runtime](../jouledb-ai-runtime/) — every cascade response carries a receipt sourced here
