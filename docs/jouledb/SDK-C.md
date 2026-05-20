# JouleDB C SDK Quickstart

**Version 1.0 — 2026-05-18**
**Crate:** [`joule-db-c`](../../crates/joule-db-c/)
**API style:** SQLite-flavoured `prepare` / `bind` / `step` / `column`

The C ABI is the path for embedding JouleDB from any language with a C FFI — Python (`ctypes` / `cffi`), Go (`cgo`), Swift, C++, Node (`N-API`), Ruby, Lua, etc.

## 1. Build

`joule-db-c` is configured as `crate-type = ["cdylib", "staticlib"]`. From the workspace root:

```bash
cargo build --release -p joule-db-c
# Output: target/release/libjoule_db_c.so (Linux), libjoule_db_c.dylib (macOS), joule_db_c.dll (Windows)
```

For a `.a` static library:

```bash
cargo build --release -p joule-db-c
# target/release/libjoule_db_c.a
```

## 2. Generate the header

There is no pre-generated `joule_db.h` in tree. Generate one via `cbindgen`:

```bash
cargo install cbindgen
cd crates/joule-db-c
cbindgen --config cbindgen.toml --crate joule-db-c --output joule_db.h
```

## 3. Minimal example

```c
#include "joule_db.h"
#include <stdio.h>

int main(void) {
    JouleDB *db = joule_db_open("my.db");
    if (!db) { fprintf(stderr, "open failed\n"); return 1; }

    joule_db_exec(db, "CREATE TABLE IF NOT EXISTS users (id INT, name TEXT)", NULL);
    joule_db_exec(db, "INSERT INTO users VALUES (1, 'Alice')", NULL);
    joule_db_exec(db, "INSERT INTO users VALUES (2, 'Bob')", NULL);

    JouleDBStmt *stmt = joule_db_prepare(db, "SELECT id, name FROM users WHERE id = $1");
    joule_db_bind_int(stmt, 1, 1);

    while (joule_db_step(stmt) == JOULE_ROW) {
        int id = joule_db_column_int(stmt, 0);
        const char *name = joule_db_column_text(stmt, 1);
        printf("%d %s\n", id, name);
    }

    joule_db_finalize(stmt);
    joule_db_close(db);
    return 0;
}
```

Compile:

```bash
clang -L target/release -ljoule_db_c -lpthread -ldl -lm example.c -o example
./example
```

## 4. Result codes

| Constant | Value | Meaning |
|---|---|---|
| `JOULE_OK` | `0` | Success |
| `JOULE_ERROR` | `1` | Generic error |
| `JOULE_ROW` | `100` | `step()` produced a row |
| `JOULE_DONE` | `101` | `step()` completed |

## 5. Bind / column families

| Bind | Column |
|---|---|
| `joule_db_bind_int(stmt, idx, c_int)` | `joule_db_column_int(stmt, col) -> c_int` |
| `joule_db_bind_double(stmt, idx, c_double)` | `joule_db_column_double(stmt, col) -> c_double` |
| `joule_db_bind_text(stmt, idx, const char *)` | `joule_db_column_text(stmt, col) -> const char *` |
| `joule_db_bind_blob(stmt, idx, const u8 *, size_t)` | `joule_db_column_blob(stmt, col, out_size) -> const u8 *` |
| `joule_db_bind_null(stmt, idx)` | `joule_db_column_is_null(stmt, col) -> int` |

Indexes are **1-based** for binds (SQLite convention), **0-based** for columns.

## 6. Energy receipts

Per-statement energy is queryable after `step` returns `JOULE_DONE`:

```c
uint64_t total_uwh = joule_db_stmt_energy_uwh(stmt);
printf("%lu µWh consumed\n", total_uwh);
```

## 7. Threading

The opaque `JouleDB` and `JouleDBStmt` handles are **not** thread-safe on their own. Wrap accesses with your own mutex, or open one `JouleDB` handle per thread (handles are cheap — they share the underlying engine).

## 8. See also

- [`crates/joule-db-c/README.md`](../../crates/joule-db-c/README.md)
- [`SDK-RUST.md`](SDK-RUST.md) — the Rust client this wraps internally
- [`SDK-ODBC.md`](SDK-ODBC.md) — for BI tooling (no FFI needed)
