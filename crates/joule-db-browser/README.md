# joule-db-browser

Browser / WASM bindings for JouleDB with IndexedDB and OPFS storage.

`joule-db-browser` runs [`joule-db-core`](../joule-db-core/) inside a browser tab. Two storage backends — IndexedDB (universal) and Origin-Private File System (modern Chromium / Safari) — back the same B-tree engine that runs on native. WebGPU acceleration is available via [`joule-db-gpu`](../joule-db-gpu/) where the browser supports it.

## Module map

| Module | Role |
|---|---|
| [`wasm.rs`](src/wasm.rs) | wasm-bindgen entry points |
| [`engine/`](src/engine/), [`engine.rs`](src/engine.rs), [`engine_enhanced.rs`](src/engine_enhanced.rs) | Engine wrapper + browser-specific extensions |
| [`storage/`](src/storage/) | IndexedDB + OPFS backends |
| [`persistence/`](src/persistence/) | Browser-specific persistence layer |
| [`gpu/`](src/gpu/) | WebGPU compute path |
| [`htap/`](src/htap/) | HTAP compression (mixed transactional + analytical workloads) |
| [`ipc/`](src/ipc/) | Cross-tab IPC via BroadcastChannel |
| [`apis/`](src/apis/) | Browser API surface (postMessage, fetch, etc.) |
| [`learned/`](src/learned/) | Learned-index variant |
| [`viz/`](src/viz/) | Visualization adapter via [`joule-db-viz`](../joule-db-viz/) |
| [`monitoring/`](src/monitoring/) | In-page telemetry |
| [`resilience/`](src/resilience/) | Browser-quirk resilience (quota errors, tab-throttle) |
| [`deterministic/`](src/deterministic/) | Deterministic-replay harness |
| [`novel/`](src/novel/) | Research primitives |

## Examples

Six in-tree examples in [`examples/`](examples/):

| Example | Demonstrates |
|---|---|
| `basic_usage.rs` | Open, insert, query |
| `error_recovery.rs` | Recover from quota / corruption errors |
| `gpu_acceleration.rs` | WebGPU compute path |
| `htap_compression_demo.rs` | HTAP compression workflow |
| `monitoring.rs` | Per-page telemetry hooks |
| `query_to_viz.rs` | Query → [`joule-db-viz`](../joule-db-viz/) → Vega-Lite chart |

## Tests

42 tests in `src/`, plus 7 integration tests in [`tests/`](tests/) covering HTAP compression, learned-index, and deterministic-replay surfaces. 2 benches in [`benches/`](benches/).

## See also

- [joule-db-core](../joule-db-core/) — the engine being run
- [joule-db-gpu](../joule-db-gpu/) — WebGPU compute
- `docs/jouledb/SDK-WASM.md` *(in progress)*
