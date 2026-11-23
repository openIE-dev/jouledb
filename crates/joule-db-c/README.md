# joule-db-c

JouleDB C Foreign Function Interface.

`joule-db-c` is the SQLite-style C API for embedding JouleDB in any language with a C ABI — Python (via `ctypes` / `cffi`), Go (`cgo`), Swift, C++, Node (`N-API`), Ruby, Lua, anywhere C bindings reach.

## API style

Follows the SQLite `prepare` / `bind` / `step` / `column` pattern:

```c
JouleDB *db = joule_db_open("my.db");
joule_db_exec(db, "CREATE TABLE users (id INT, name TEXT)", NULL);

JouleDBStmt *stmt = joule_db_prepare(db, "SELECT * FROM users WHERE id = $1");
joule_db_bind_int(stmt, 1, 42);
while (joule_db_step(stmt) == JOULE_ROW) {
    int id = joule_db_column_int(stmt, 0);
    const char *name = joule_db_column_text(stmt, 1);
}
joule_db_finalize(stmt);
joule_db_close(db);
```

## Result codes

| Constant | Value | Meaning |
|---|---|---|
| `JOULE_OK` | `0` | Success |
| `JOULE_ERROR` | `1` | Generic error |
| `JOULE_ROW` | `100` | `step()` produced a row |
| `JOULE_DONE` | `101` | `step()` completed (no more rows) |

## Build

Build the crate as a `cdylib` / `staticlib` (configured in [`Cargo.toml`](Cargo.toml)). Header generation is downstream — there is no in-tree `joule_db.h` yet; consumers should generate one via `cbindgen` against the public symbols in [`src/lib.rs`](src/lib.rs).

## Underlying engine

The C ABI wraps [`joule-db-local`](../joule-db-local/) (disk-backed) and dispatches queries through [`joule-db-query`](../joule-db-query/)'s `StorageExecutor`.

## See also

- [joule-db-odbc](../joule-db-odbc/) — for BI-tool clients
- [joule-db-client](../joule-db-client/) — for Rust callers
- `docs/jouledb/SDK-C.md` *(in progress)* — header reference + linking guide
