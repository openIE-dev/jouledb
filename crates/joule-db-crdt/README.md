# joule-db-crdt

Conflict-free replicated data types for JouleDB edge sync and disconnected ops.

CRDTs let multiple replicas accept writes independently — phones / laptops / edge devices going on and offline — and merge without coordination. `joule-db-crdt` is the thin CRDT layer that powers JouleDB's federated and disconnected deployments.

## Module map

| Module | Role |
|---|---|
| [`types.rs`](src/types.rs) | CRDT type definitions (G-Counter, PN-Counter, OR-Set, LWW-Register, etc.) |

## Tests

15 tests in `src/`.

## See also

- [joule-db-edge](../joule-db-edge/) — uses CRDTs for federated sync
- [joule-db-branch](../joule-db-branch/) — sibling crate for CoW branching
- [joule-db-server::replication](../joule-db-server/src/replication.rs)
