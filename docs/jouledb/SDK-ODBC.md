# JouleDB ODBC Driver Quickstart

**Version 1.0 — 2026-05-18**
**Crate:** [`joule-db-odbc`](../../crates/joule-db-odbc/) (73 tests)
**Spec:** ODBC 3.x
**Server backends spoken:** JWP and pgwire

The ODBC driver lets Excel, Tableau, Power BI, DBeaver, Looker, and anything else that speaks ODBC connect to a JouleDB cluster without code changes. Behind the scenes the driver talks pgwire to [`joule-db-server`](../../crates/joule-db-server/).

## 1. Build

```bash
cargo build --release -p joule-db-odbc
# Output: target/release/libjoule_db_odbc.{so,dylib,dll}
```

Install:

| OS | Path |
|---|---|
| Linux | `/usr/lib/x86_64-linux-gnu/odbc/libjouledb.so` |
| macOS | `/Library/ODBC/JouleDB.bundle/Contents/MacOS/libjouledb.dylib` |
| Windows | `%SystemRoot%\System32\jouledb.dll` |

## 2. Register the driver

### Linux (`/etc/odbcinst.ini`)

```ini
[JouleDB]
Description = JouleDB ODBC Driver
Driver      = /usr/lib/x86_64-linux-gnu/odbc/libjouledb.so
Setup       = /usr/lib/x86_64-linux-gnu/odbc/libjouledb.so
UsageCount  = 1
```

### macOS (via `iodbcadm-gtk` or `/Library/ODBC/odbcinst.ini`)

Same content as Linux.

### Windows

Use **ODBC Data Sources (64-bit)** in Control Panel → Drivers tab → Add → point to `jouledb.dll`.

## 3. Define a DSN

### Linux / macOS (`~/.odbc.ini`)

```ini
[my-cluster]
Description = JouleDB cluster — startup tier
Driver      = JouleDB
Server      = my-cluster.jouledb.cloud
Port        = 9000
Database    = main
UID         = my-user
PWD         = my-password
TLS         = yes
EnergyBudgetUJ = 50000000   # 50 J per query cap
```

### Windows

Open ODBC Data Source Administrator → System DSN → Add → JouleDB → fill in the same fields.

## 4. Test the connection

```bash
isql -v my-cluster
SQL> SELECT 1 + 1;
+------+
| ?    |
+------+
| 2    |
+------+
```

## 5. Connect from BI tools

### Excel

Data → Get Data → From Other Sources → From ODBC → pick `my-cluster` DSN → enter credentials. Tables show up as Excel tables.

### Tableau

Connect → To a Server → Other Databases (ODBC) → pick `my-cluster` DSN. JouleDB's pgwire surface advertises PostgreSQL-compatible metadata so Tableau's metadata discovery works out of the box.

### Power BI

Get Data → ODBC → pick the DSN. Use Import or DirectQuery mode.

### DBeaver

New Connection → ODBC → DSN: `my-cluster`. Configure JDBC properties as needed.

## 6. Energy receipts via ODBC

The driver exposes the per-statement energy total through a custom attribute. Most BI tools won't display it directly, but you can query it explicitly:

```sql
SELECT energy_receipt() AS uwh FROM (SELECT * FROM expensive_query) AS x LIMIT 0;
```

Or set a per-connection budget via DSN attribute `EnergyBudgetUJ`. The server rejects queries projected to exceed it; the BI tool sees a structured error.

## 7. Supported SQL surface

Per [`QUERY-SQL.md`](QUERY-SQL.md). The ODBC driver passes SQL through to the server unchanged — windowing, CTEs, joins, set ops, vector ops, JSON ops all work.

## 8. Type mapping

| JouleDB type | ODBC type | Notes |
|---|---|---|
| `INTEGER` | `SQL_INTEGER` | |
| `BIGINT` | `SQL_BIGINT` | |
| `REAL` | `SQL_REAL` | |
| `DOUBLE PRECISION` | `SQL_DOUBLE` | |
| `NUMERIC(p, s)` | `SQL_NUMERIC` | |
| `TEXT` | `SQL_VARCHAR` | unbounded; clients may report as `MAX` |
| `BOOLEAN` | `SQL_BIT` | |
| `BYTEA` | `SQL_VARBINARY` | |
| `DATE` | `SQL_TYPE_DATE` | |
| `TIMESTAMP` | `SQL_TYPE_TIMESTAMP` | |
| `JSON` / `JSONB` | `SQL_LONGVARCHAR` | serialized as text |
| `VECTOR(n)` | `SQL_VARBINARY` | binary-encoded |

## 9. See also

- [`crates/joule-db-odbc/README.md`](../../crates/joule-db-odbc/README.md)
- [`QUERY-SQL.md`](QUERY-SQL.md) — the SQL surface ODBC clients see
- [`crates/joule-db-server/src/pgwire.rs`](../../crates/joule-db-server/src/pgwire.rs) — the pgwire backend
