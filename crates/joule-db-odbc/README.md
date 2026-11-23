# joule-db-odbc

ODBC 3.x driver for JouleDB — connects Excel, Tableau, Power BI, and any ODBC-compatible tool.

Drop-in driver for the world's BI tooling. Speaks ODBC 3.x to the client; speaks JWP or pgwire to a [`joule-db-server`](../joule-db-server/) deployment on the back end.

## Module map

| Module | Role |
|---|---|
| [`api.rs`](src/api.rs) | ODBC API entry points (`SQLAllocHandle`, `SQLConnect`, `SQLExecDirect`, …) |
| [`handle.rs`](src/handle.rs) | Handle lifecycle — environment, connection, statement, descriptor |
| [`protocol.rs`](src/protocol.rs) | JouleDB protocol bridge (JWP / pgwire) |
| [`types.rs`](src/types.rs) | ODBC type ↔ JouleDB type mapping |

## Tests

73 tests in `src/`.

## See also

- [joule-db-server](../joule-db-server/) — the daemon this driver talks to (`pgwire.rs`)
- `docs/jouledb/SDK-ODBC.md` *(in progress)* — DSN config, Excel / Tableau / Power BI setup
