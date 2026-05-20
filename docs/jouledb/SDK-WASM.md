# JouleDB Browser / WASM SDK Quickstart

**Version 1.0 — 2026-05-18**
**Crate:** [`joule-db-browser`](../../crates/joule-db-browser/) (42 tests, 6 examples)
**Storage backends:** IndexedDB (universal), OPFS (Origin-Private File System)
**Compute backend:** WebGPU (via [`joule-db-gpu`](../../crates/joule-db-gpu/))

JouleDB runs **inside the browser**. Same B-tree engine as native, same query languages, persistent across page reloads via IndexedDB or OPFS, GPU-accelerated where the browser supports it.

## 1. Build

```bash
cd crates/joule-db-browser
wasm-pack build --release --target web
# Output: pkg/joule_db_browser_bg.wasm + pkg/joule_db_browser.js
```

## 2. Use from JavaScript

```html
<script type="module">
  import init, { Database } from './pkg/joule_db_browser.js';

  await init();

  // Choose backend: 'idb' (IndexedDB) or 'opfs' (OPFS)
  const db = await Database.open('my-app', { backend: 'idb' });

  await db.exec("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, body TEXT)");
  await db.exec("INSERT INTO notes (body) VALUES ('hello world')");

  const rows = await db.query("SELECT * FROM notes");
  console.log(rows);

  console.log(`energy: ${rows.energy_uwh} µWh`);
</script>
```

## 3. From a TypeScript / React project

```ts
import init, { Database } from 'joule-db-browser';

let db: Database;

useEffect(() => {
  init().then(async () => {
    db = await Database.open('my-app', { backend: 'opfs' });
    setReady(true);
  });
}, []);

const search = async (query: string) => {
  const rows = await db.query(
    "SELECT id, body FROM notes WHERE body LIKE $1 LIMIT 20",
    [`%${query}%`]
  );
  setResults(rows);
};
```

## 4. WebGPU acceleration

Enable WebGPU compute (where supported — Chromium 113+, Safari Tech Preview, Firefox Nightly with flag):

```js
const db = await Database.open('my-app', {
  backend: 'opfs',
  gpu: true,
});

// Vector k-NN now runs on the GPU
const neighbors = await db.query(
  "SELECT id, title FROM articles ORDER BY embedding <-> $1 LIMIT 10",
  [queryEmbedding]
);
```

## 5. HTAP compression

For mixed transactional + analytical workloads in-browser, [`joule-db-browser::htap`](../../crates/joule-db-browser/src/htap/) offers compression that adapts to the read/write ratio:

```js
const db = await Database.open('my-app', {
  backend: 'opfs',
  htap: true,
});
```

## 6. Cross-tab IPC

The browser crate supports cross-tab broadcast via `BroadcastChannel` ([`ipc/`](../../crates/joule-db-browser/src/ipc/)) so a write in one tab is observable in another:

```js
db.subscribe('notes', (change) => {
  console.log('change:', change);
});
```

## 7. Examples in tree

Six runnable examples in [`crates/joule-db-browser/examples/`](../../crates/joule-db-browser/examples/):

| Example | Demonstrates |
|---|---|
| `basic_usage.rs` | Open, insert, query |
| `error_recovery.rs` | Recover from quota / corruption errors |
| `gpu_acceleration.rs` | WebGPU compute path |
| `htap_compression_demo.rs` | HTAP workflow |
| `monitoring.rs` | Per-page telemetry |
| `query_to_viz.rs` | Query → [`joule-db-viz`](../../crates/joule-db-viz/) → Vega-Lite chart |

## 8. Persistence guarantees

| Backend | Browser support | Capacity | Notes |
|---|---|---|---|
| `idb` (IndexedDB) | Universal | ~50% of free disk by default | Asynchronous; eviction possible under quota pressure |
| `opfs` (OPFS) | Chromium 102+, Safari 15.2+, Firefox 111+ | ~10 GB+ | Synchronous file access; survives quota pressure better |

For mission-critical persistence, prefer OPFS and call [`navigator.storage.persist()`](https://developer.mozilla.org/en-US/docs/Web/API/StorageManager/persist) to request "persistent" status from the browser.

## 9. See also

- [`crates/joule-db-browser/README.md`](../../crates/joule-db-browser/README.md)
- [`crates/joule-db-gpu/README.md`](../../crates/joule-db-gpu/README.md)
- [`SDK-RUST.md`](SDK-RUST.md) — the native equivalent
