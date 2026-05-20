# JouleDB CQL Reference

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-db-query/src/cql.rs`](../../crates/joule-db-query/src/cql.rs) (1,301 LOC)
**Executor:** [`crates/joule-db-server/src/cql_executor.rs`](../../crates/joule-db-server/src/cql_executor.rs)
**Sister docs:** [`QUERY-SQL.md`](QUERY-SQL.md)

JouleDB CQL is a subset of Cassandra Query Language for wide-column workloads. The parser and executor were hardened in May 2026 to fix two long-standing gaps (comparison operators and aggregates) — see [`crates/joule-db-query/README.md`](../../crates/joule-db-query/README.md) §"Recent fixes."

## 1. Supported

| Surface | Status |
|---|---|
| `SELECT`, `INSERT`, `UPDATE`, `DELETE` | yes |
| `CREATE KEYSPACE`, `USE` | yes |
| `CREATE TABLE` with composite partition + clustering keys | yes |
| `WHERE` with `=`, `>`, `<`, `>=`, `<=`, `!=`, `IN` | yes (was `=` and `IN` only before May 2026) |
| Aggregates: `COUNT(*)`, `COUNT(col)`, `SUM(col)`, `AVG(col)`, `MIN(col)`, `MAX(col)` | yes (was unimplemented before May 2026) |
| `ALLOW FILTERING` | yes (no-op — JouleDB doesn't enforce Cassandra's filtering restrictions) |
| `TTL` and `USING TIMESTAMP` | partial |
| Lightweight transactions (`IF NOT EXISTS`, `IF`) | yes |
| Counters | yes |
| Collections (`list`, `set`, `map`) | yes |
| User-defined types (UDT) | yes |
| Materialized views | partial |
| Secondary indexes | yes |

## 2. Examples

```cql
-- Keyspace + table with composite key
CREATE KEYSPACE shop WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
USE shop;

CREATE TABLE products (
    category    text,
    sku         text,
    name        text,
    price       decimal,
    PRIMARY KEY ((category), sku)
);

-- Comparison operators
SELECT * FROM products WHERE category = 'electronics' AND price > 100 ALLOW FILTERING;

-- Aggregates
SELECT category, COUNT(*), AVG(price), MAX(price)
FROM products
GROUP BY category;

-- Counter column
CREATE TABLE page_views (page text PRIMARY KEY, views counter);
UPDATE page_views SET views = views + 1 WHERE page = '/home';
```

## 3. Where it differs from Cassandra

- **Single substrate** — there is no Cassandra-style partition routing. `ALLOW FILTERING` is accepted but JouleDB just executes the query.
- **Replication factor** — declared but enforcement is via Raft replicas on the server, not per-keyspace.
- **CQL → standard executor** — both paths execute through [`joule-db-query::execution`](../../crates/joule-db-query/src/execution.rs); behaviour matches SQL for shared operators.

## 4. See also

- [`crates/joule-db-query/README.md`](../../crates/joule-db-query/README.md) — recent CQL parser/executor fixes
- [`QUERY-SQL.md`](QUERY-SQL.md) — the equivalent SQL surface
