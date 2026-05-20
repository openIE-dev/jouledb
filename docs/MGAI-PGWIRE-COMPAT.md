# MGAI pgwire Compatibility Reference

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-db-server/src/pgwire.rs`](../crates/joule-db-server/src/pgwire.rs) (~135K LOC)
**Wire:** PostgreSQL Frontend/Backend Protocol v3
**Default port:** 5432 (when enabled)

JouleDB speaks **PostgreSQL v3 wire protocol** as an alternate transport to JWP. This is what lets `psql`, `pg_dump`, JDBC, ODBC, SQLAlchemy, ActiveRecord, Diesel, Tokio Postgres, and every other PostgreSQL-compatible client connect without code changes.

---

## 1. What's wired up

| Surface | Status |
|---|---|
| Startup message + parameter negotiation | yes |
| Authentication: SCRAM-SHA-256 | yes |
| Authentication: SASL channel binding | yes |
| Authentication: trust / password | yes (for local-dev) |
| Authentication: cert | yes (when TLS is on) |
| Simple query protocol | yes |
| Extended query protocol (parse / bind / execute / sync) | yes |
| Prepared statements (named + unnamed) | yes |
| Portals (cursors over executed statements) | yes |
| `SET` parameters | yes |
| Cancel request | yes |
| Notifications (`LISTEN` / `NOTIFY`) | partial — translates to JWP subscriptions |
| Copy in / out (`COPY ... FROM STDIN`, `COPY ... TO STDOUT`) | yes (text + binary format) |
| `pg_catalog` metadata tables | yes (6 tables — `pg_type`, `pg_namespace`, `pg_class`, `pg_attribute`, `pg_database`, `pg_settings`) |
| Type OIDs | 24 aliases — see [`QUERY-SQL.md`](jouledb/QUERY-SQL.md) §2 |
| TLS / SSL handshake | yes (when `tls` feature is on) |

---

## 2. Connect with `psql`

```bash
psql -h my-cluster.jouledb.cloud -p 5432 -U my-user -d main
```

```text
psql (16.0)
SSL connection (protocol: TLSv1.3, cipher: TLS_AES_256_GCM_SHA384)
Type "help" for help.

main=> SELECT version();
                              version
-------------------------------------------------------------------
 JouleDB 0.2 (compatible with PostgreSQL 15.0)
(1 row)

main=> \d users
                         Table "public.users"
  Column   |  Type   | Collation | Nullable |       Default
-----------+---------+-----------+----------+--------------------
 id        | bigint  |           | not null | nextval(...)
 name      | text    |           |          |
 age       | integer |           |          |
```

`\d`, `\dt`, `\du`, `\df`, `\dn` all work — they're standard SQL queries against `pg_catalog`.

---

## 3. JDBC

```java
String url = "jdbc:postgresql://my-cluster.jouledb.cloud:5432/main";
Properties props = new Properties();
props.setProperty("user", "my-user");
props.setProperty("password", "my-password");
props.setProperty("ssl", "true");
props.setProperty("sslmode", "require");

Connection conn = DriverManager.getConnection(url, props);
PreparedStatement stmt = conn.prepareStatement("SELECT * FROM users WHERE id = ?");
stmt.setLong(1, 42);
ResultSet rs = stmt.executeQuery();
```

---

## 4. Python — psycopg / asyncpg

```python
# psycopg 3
import psycopg
conn = psycopg.connect("postgresql://my-user@my-cluster.jouledb.cloud:5432/main?sslmode=require")
with conn.cursor() as cur:
    cur.execute("SELECT * FROM users WHERE id = %s", (42,))
    for row in cur:
        print(row)
```

```python
# asyncpg
import asyncpg
async def main():
    conn = await asyncpg.connect("postgresql://my-user@my-cluster.jouledb.cloud:5432/main")
    rows = await conn.fetch("SELECT * FROM users WHERE active = true")
    for row in rows:
        print(row["name"])
```

---

## 5. SQLAlchemy

```python
from sqlalchemy import create_engine
engine = create_engine("postgresql+psycopg://user:pass@my-cluster.jouledb.cloud:5432/main")
```

Drop-in replacement; ORM models work unchanged.

---

## 6. Differences from PostgreSQL — by-design

| Behaviour | PostgreSQL | JouleDB |
|---|---|---|
| Unknown column reference | could silently return NULL | **errors** with `COLUMN_NOT_FOUND` (May 2026 fix) |
| Energy receipts | n/a | every query emits per-stage joules; surface as `EXPLAIN` annotation or `extensions.energy` in GraphQL |
| `LISTEN` / `NOTIFY` | server-side broadcast | translated to JWP subscriptions; semantics similar but wire detail differs |
| Stored procedures | `LANGUAGE plpgsql` etc. | `LANGUAGE wasm` only (via the `wasm-triggers` feature) |
| Extensions (`CREATE EXTENSION`) | rich ecosystem | JouleDB-native: `vector`, `hdc`, `timeseries`, `langgraph`, `jwp` — all built-in, not extensions |
| Replication slots | physical / logical | JouleDB uses Raft or read-replica WAL streaming instead |
| pgvector | extension | built-in (see [`QUERY-SQL.md`](jouledb/QUERY-SQL.md) §6) |
| Foreign data wrappers (FDW) | rich ecosystem | partial — sigql / federation surface (see [`crates/sigql/`](../crates/sigql/)) |

---

## 7. Differences not by design (gaps)

These exist because pgwire is a partial surface, not a bug-for-bug clone:

- **`pg_hba.conf` semantics** — JouleDB auth is configured in `config.toml`, not pg_hba.
- **Multiple databases per server** — JouleDB has one database per server; the `dbname` parameter is accepted for compatibility but ignored.
- **Schemas** — supported but not heavily exercised; `public` is the default.
- **Sequence semantics under load** — `SERIAL` / `BIGSERIAL` work but `nextval` under contention has not been stress-tested at PostgreSQL parity.
- **System columns** (`ctid`, `xmin`, `xmax`) — partially exposed; check the source if you depend on them.

For PostgreSQL features that *aren't* implemented, the server returns a structured error with a fallback suggestion (e.g., "use `LANGUAGE wasm` instead of `LANGUAGE plpgsql`").

---

## 8. Authentication

### 8.1 SCRAM-SHA-256 (default)

Implementation: [`crates/joule-db-server/src/scram.rs`](../crates/joule-db-server/src/scram.rs). Standards-compliant; works with any modern PostgreSQL client.

### 8.2 Trust auth (local-dev)

```toml
# config.toml
[pgwire.auth]
mode = "trust"
```

Anyone can connect as any role without a password. **Don't** ship this to production.

### 8.3 Cert auth

```toml
[pgwire.tls]
enabled = true
cert_file = "/etc/jouledb/tls.crt"
key_file = "/etc/jouledb/tls.key"
ca_file = "/etc/jouledb/ca.crt"
require_client_cert = true
```

---

## 9. Performance characteristics

| Workload | JouleDB pgwire vs. PostgreSQL |
|---|---|
| Single-row PK lookup | comparable (both index-scan a B-tree) |
| Sequential scan | comparable |
| OLAP aggregates | JouleDB faster with `arrow-execution` or `datafusion-backend` features enabled |
| Vector k-NN | JouleDB faster (native `vector` type, no `pgvector` extension overhead) |
| HDC similarity | JouleDB-only |
| Recursive CTEs | comparable |
| Many concurrent connections | JouleDB has a smaller per-connection overhead (no fork-per-backend model) |
| Replication | JouleDB Raft vs. PostgreSQL streaming replication — see [`RUNBOOK-CLUSTERING.md`](jouledb/RUNBOOK-CLUSTERING.md) |

---

## 10. See also

- [`MGAI-JWP-PROTOCOL.md`](MGAI-JWP-PROTOCOL.md) — JouleDB-native wire (energy-aware, agent-contract-aware)
- [`QUERY-SQL.md`](jouledb/QUERY-SQL.md) — the SQL surface exposed via pgwire
- [`SDK-ODBC.md`](jouledb/SDK-ODBC.md) — BI tooling
- [`crates/joule-db-server/src/pgwire.rs`](../crates/joule-db-server/src/pgwire.rs) — source

---

*Drafted 2026-05-18 as wave 3 of the JouleDB documentation parity pass.*
