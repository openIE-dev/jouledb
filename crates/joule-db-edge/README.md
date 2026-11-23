# joule-db-edge

JouleDB for edge devices with hardware acceleration (GPU / NPU / TPU).

`joule-db-edge` is the deployment target for ARM SBCs, NPUs, TPUs, and IoT hardware. It wraps [`joule-db-core`](../joule-db-core/) with hardware-acceleration abstractions, edge-specific memory management, and sync primitives for federated / disconnected deployments.

## Module map

| Module | Role |
|---|---|
| [`database.rs`](src/database.rs) | Edge-deployment database wrapper |
| [`accelerator.rs`](src/accelerator.rs) | GPU / NPU / TPU accelerator abstraction |
| [`backends/`](src/backends/) | Hardware-specific backends |
| [`memory.rs`](src/memory.rs) | Edge-aware memory budgeting |
| [`storage.rs`](src/storage.rs) | Edge storage layer |
| [`sync.rs`](src/sync.rs) | Federated sync / CRDT bridge |
| [`config.rs`](src/config.rs) | Edge config |

## Tests

81 tests in `src/`.

## See also

- [joule-db-core](../joule-db-core/)
- [joule-db-gpu](../joule-db-gpu/) — wgpu compute backend
- [joule-db-crdt](../joule-db-crdt/) — CRDTs for edge sync
