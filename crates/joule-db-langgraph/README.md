# joule-db-langgraph

LangGraph integration for JouleDB — checkpoint and message stores with semantic search.

`joule-db-langgraph` is the JouleDB-backed implementation of LangGraph's checkpoint and message-store interfaces. LangGraph agents that pick this backend get B-tree-durable checkpoints, HDC-powered semantic search over message history, and joule receipts on every read/write.

## Module map

| Module | Role |
|---|---|
| [`checkpoint.rs`](src/checkpoint.rs) | LangGraph `Checkpointer` impl — graph state snapshots |
| [`messages.rs`](src/messages.rs) | LangGraph message store — conversation history with semantic search |
| [`error.rs`](src/error.rs) | Error types |

## Tests

16 tests in `src/`.

## Server bridge

[`joule-db-server::langgraph_handlers`](../joule-db-server/src/langgraph_handlers.rs) exposes this crate's surface over the wire.

## See also

- [joule-db-core](../joule-db-core/) — the storage backend
- [joule-db-hdc](../joule-db-hdc/) — semantic search substrate
- `docs/jouledb/SDK-LANGGRAPH.md` *(in progress)*
