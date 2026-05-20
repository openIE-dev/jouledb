# joule-db-branch

Copy-on-write database branching with energy budgets for JouleDB.

Git for databases. `joule-db-branch` lets you fork a JouleDB database in O(1), make experimental writes on the branch, measure their energy cost separately, and merge or discard without ever touching the parent. Branches share unmodified pages via copy-on-write — disk cost grows only with divergence.

## Module map

| Module | Role |
|---|---|
| [`manager.rs`](src/manager.rs) | Branch manager — create, list, merge, drop |
| [`storage.rs`](src/storage.rs) | CoW page sharing + per-branch overlay |
| [`energy.rs`](src/energy.rs) | Per-branch joule accounting + budget enforcement |

## Tests

21 tests in `src/`.

## See also

- [joule-db-core](../joule-db-core/) — the underlying CoW MVCC engine
- [joule-db-energy](../joule-db-energy/) — joule accounting primitives
