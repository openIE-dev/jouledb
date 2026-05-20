# JouleDB LangGraph Integration Quickstart

**Version 1.0 — 2026-05-18**
**Crate:** [`joule-db-langgraph`](../../crates/joule-db-langgraph/) (16 tests)
**Server bridge:** [`joule-db-server::langgraph_handlers`](../../crates/joule-db-server/src/langgraph_handlers.rs)

`joule-db-langgraph` is the JouleDB-backed implementation of LangGraph's `Checkpointer` and message-store interfaces. LangGraph agents that pick this backend get B-tree-durable checkpoints, HDC-powered semantic search over message history (via [`joule-db-hdc`](../../crates/joule-db-hdc/)), and joule receipts on every read / write.

## 1. Use it from Rust LangGraph

```rust
use joule_db_langgraph::{JouleDBCheckpointer, JouleDBMessageStore};
use joule_db_core::Database;

let db = Database::open("/var/lib/langgraph.db")?;

let checkpointer = JouleDBCheckpointer::new(db.clone());
let messages = JouleDBMessageStore::new(db.clone());

// Pass to the LangGraph builder
let graph = MyGraph::builder()
    .checkpointer(checkpointer)
    .message_store(messages)
    .build();
```

## 2. Checkpoint store

```rust
use joule_db_langgraph::Checkpointer;

// Write a checkpoint
checkpointer.put_checkpoint("thread-123", state).await?;

// Read latest
let state = checkpointer.get_checkpoint("thread-123").await?;

// List history
let versions = checkpointer.list_checkpoints("thread-123").await?;
```

Internally each checkpoint is stored as a `(thread_id, version_id) → state` row in JouleDB with the state body in extent-allocated overflow pages. Versions are monotonic; rollback is a `get_checkpoint_at(thread, version_id)` call.

## 3. Message store with semantic search

```rust
use joule_db_langgraph::MessageStore;

// Append
messages.append("thread-123", Message {
    role: "user",
    content: "What did we decide about the Q3 timeline?",
}).await?;

// Read recent
let recent = messages.recent("thread-123", 10).await?;

// Semantic search — uses joule-db-hdc under the hood
let matches = messages.search("thread-123", "timeline decisions", 5).await?;
for m in matches {
    println!("{}: {}", m.score, m.content);
}
```

## 4. Energy receipts

Every checkpoint write and message-store operation returns an `EnergyReceipt`:

```rust
let receipt = checkpointer.put_checkpoint("thread-123", state).await?;
println!("checkpoint cost: {} µJ at tier {:?}", receipt.joules_uj, receipt.tier);
```

Semantic search is HDC-backed — typical cost is µJ-class regardless of corpus size (vs. embedding-model search which scales with the model).

## 5. From Python LangGraph

A thin Python wrapper around the Rust crate lives in `bindings/python/jouledb_langgraph/` *(roadmap — not yet shipped)*. For now, Python callers can hit the server's `langgraph_handlers` endpoint over HTTP:

```python
import requests

# Checkpoint write
r = requests.post(
    "http://localhost:8080/langgraph/checkpoint",
    json={"thread_id": "thread-123", "state": {...}},
    headers={"Authorization": "Bearer ..."},
)
print(r.headers["X-Energy-uJ"])
```

## 6. See also

- [`crates/joule-db-langgraph/README.md`](../../crates/joule-db-langgraph/README.md)
- [`MGAI-HDC-REFERENCE.md`](../MGAI-HDC-REFERENCE.md) — the HDC primitives that power message search
- [`SDK-RUST.md`](SDK-RUST.md) — the Rust client SDK
