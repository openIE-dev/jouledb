# joule-db-client

Rust client SDK for JouleDB — the energy-aware multi-model database.

`joule-db-client` is the async Rust client for talking to a [`joule-db-server`](../joule-db-server/) deployment over JWP, pgwire, HTTP, or WebSocket. Connection pooling, retries, energy receipt parsing.

## Module map

| Module | Role |
|---|---|
| [`client.rs`](src/client.rs) | Top-level `Client` type |
| [`connection.rs`](src/connection.rs) | Single connection lifecycle |
| [`pool.rs`](src/pool.rs) | Connection pool |
| [`protocol.rs`](src/protocol.rs) | Wire framing (JWP / pgwire) |
| [`error.rs`](src/error.rs) | Error types |

## Tests

51 tests in `src/`.

## See also

- [joule-db-server](../joule-db-server/) — the server side
- [joule-db-c](../joule-db-c/) — the C ABI for non-Rust clients
- [joule-db-odbc](../joule-db-odbc/) — the ODBC driver for BI tools
- [joule-db-browser](../joule-db-browser/) — for WASM / browser callers
- `docs/jouledb/SDK-RUST.md` *(in progress)*
