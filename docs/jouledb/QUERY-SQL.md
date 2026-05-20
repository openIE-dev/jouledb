# JouleDB SQL Reference

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-db-query/src/sql.rs`](../../crates/joule-db-query/src/sql.rs) (7,039 LOC)
**Wire:** [`pgwire.rs`](../../crates/joule-db-server/src/pgwire.rs) (PostgreSQL v3 wire), JWP `Query` frame
**Sister docs:** [`QUERY-CYPHER.md`](QUERY-CYPHER.md), [`QUERY-CQL.md`](QUERY-CQL.md), [`QUERY-GRAPHQL.md`](QUERY-GRAPHQL.md), [`QUERY-DATALOG.md`](QUERY-DATALOG.md), [`QUERY-SPARQL.md`](QUERY-SPARQL.md), [`QUERY-GREMLIN.md`](QUERY-GREMLIN.md), [`QUERY-TIMESERIES.md`](QUERY-TIMESERIES.md)

---

## 1. What we support

JouleDB SQL is an ANSI-style subset wired through one execution plan over [`joule-db-core`](../../crates/joule-db-core/). Compatible with most clients via the [pgwire](../../crates/joule-db-server/src/pgwire.rs) surface (SCRAM-SHA-256 auth, prepared statements, type-OID portals).

| Surface | Status |
|---|---|
| `SELECT` / `FROM` / `WHERE` / `GROUP BY` / `HAVING` / `ORDER BY` / `LIMIT` / `OFFSET` | yes |
| `INSERT` / `UPDATE` / `DELETE` | yes |
| `CREATE TABLE` / `DROP TABLE` / `ALTER TABLE` | yes |
| Joins: `INNER` / `LEFT` / `RIGHT` / `FULL OUTER` / `CROSS` | yes |
| Set ops: `UNION` / `UNION ALL` / `INTERSECT` / `EXCEPT` | yes |
| CTEs (`WITH name AS (…)`) — also in set-op RHS | yes |
| Subqueries (scalar, `IN`, `EXISTS`, `ANY`, `ALL`) | yes |
| `CASE WHEN … THEN … ELSE … END` | yes |
| Window functions (see §3) | yes (full set) |
| Aggregates: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `STDDEV`, `STRING_AGG`, `ARRAY_AGG` | yes |
| `DISTINCT` and `SUM(DISTINCT x)` etc. | yes |
| `GROUP BY` ordinals (`GROUP BY 1, 2`) | yes |
| `GROUP BY CASE` expressions | yes |
| Table-level `PRIMARY KEY (col1, col2)` | yes |
| `PRIMARY KEY` column constraints | yes |
| Foreign-key `REFERENCES` | parsed; non-strict enforcement |
| Constraints: `NOT NULL`, `UNIQUE`, `CHECK` | yes |
| Indexes: `CREATE INDEX`, `CREATE UNIQUE INDEX` | yes |
| `EXPLAIN` / `EXPLAIN ANALYZE` | yes (with energy receipts) |
| Transactions: `BEGIN` / `COMMIT` / `ROLLBACK` | yes |
| Savepoints: `SAVEPOINT` / `ROLLBACK TO` / `RELEASE` | yes |
| Reserved words as identifiers (`threshold`, `key`, `similar`, `meaning`, `nearest`, `primary`) | yes |
| Date tokens as identifiers (`year`, `month`, `day`, `hour`, `minute`, `second`) | yes |
| `pg_catalog` virtual tables (`pg_type`, `pg_namespace`, `pg_class`, `pg_attribute`, `pg_database`, `pg_settings`) | yes |

## 2. Type system

| Type | Aliases | Wire OID |
|---|---|---|
| `INTEGER` | `INT`, `INT4` | 23 |
| `BIGINT` | `INT8` | 20 |
| `SMALLINT` | `INT2` | 21 |
| `REAL` | `FLOAT4` | 700 |
| `DOUBLE PRECISION` | `FLOAT8` | 701 |
| `NUMERIC(p, s)` | `DECIMAL` | 1700 |
| `TEXT` | `VARCHAR`, `CHAR(n)` | 25 / 1043 / 1042 |
| `BOOLEAN` | `BOOL` | 16 |
| `BYTEA` | (binary) | 17 |
| `DATE` | — | 1082 |
| `TIME` | — | 1083 |
| `TIMESTAMP` | `TIMESTAMPTZ` | 1114 / 1184 |
| `INTERVAL` | — | 1186 |
| `JSON` / `JSONB` | — | 114 / 3802 |
| `UUID` | — | 2950 |
| `VECTOR(n)` | — | (jouledb-specific) |
| `HYPERVECTOR(n)` | — | (jouledb-specific) |

24 type-name aliases total. Vector / hypervector types are JouleDB-native; pgwire clients see them as `bytea` with a custom-type extension when probed.

## 3. Window functions

All standard SQL window functions. Frame specifications: `ROWS BETWEEN` / `RANGE BETWEEN` / `GROUPS BETWEEN`; `UNBOUNDED PRECEDING/FOLLOWING`, `CURRENT ROW`, `<n> PRECEDING/FOLLOWING`.

| Function | Behaviour |
|---|---|
| `ROW_NUMBER()` | Unique row index within partition |
| `RANK()`, `DENSE_RANK()` | Standard ranking with / without gaps |
| `PERCENT_RANK()`, `CUME_DIST()` | Statistical ranking |
| `NTILE(n)` | Bucket assignment |
| `LAG(expr, offset, default)` | Lag with explicit default (incl. negative numerics via `Unary { Neg, Literal }`) |
| `LEAD(expr, offset, default)` | Lead with explicit default |
| `FIRST_VALUE(expr)`, `LAST_VALUE(expr)`, `NTH_VALUE(expr, n)` | Frame-aware positional value |
| `SUM`, `AVG`, `COUNT`, `MIN`, `MAX` `OVER (…)` | Windowed aggregates |
| Nested in expressions: `score - AVG(score) OVER ()` | yes |

## 4. JSON ops

```sql
SELECT data->'user'->>'name' FROM events;
SELECT data #> '{a, b, c}' FROM events;
SELECT jsonb_path_exists(data, '$.user.age');
```

Operators: `->`, `->>`, `#>`, `#>>`, `?`, `?|`, `?&`, `@>`, `<@`. Functions: `jsonb_path_exists`, `jsonb_path_query`, `jsonb_set`, `jsonb_array_elements`, `jsonb_each`, `jsonb_object_keys`.

## 5. Energy receipts via `EXPLAIN`

Every `EXPLAIN` includes per-stage joule cost:

```sql
EXPLAIN SELECT u.name, COUNT(*)
        FROM users u JOIN orders o ON u.id = o.user_id
        GROUP BY u.name;
```

Returns a plan tree with a `cascade_tier` column alongside `operation`, `estimated_cost`, and `estimated_rows`. The tier is one of **Lookup** (single-row index/PK fetch, ~0.1 µJ), **Formula** (closed-form, no I/O), **Extract** (table/index scan), **Aggregate** (joins, aggregations, sorts, window functions, set ops), or **Reason** (recursive CTEs, multi-stage iteration). The query's tier is the **max** over its plan tree — the most expensive operation determines the energy class.

```sql
EXPLAIN SELECT * FROM users WHERE id = 42;
-- operation        | estimated_cost | estimated_rows | cascade_tier
-- Scan { ... }     | 1.0            | 1              | Lookup

EXPLAIN SELECT dept, COUNT(*) FROM employees GROUP BY dept;
-- operation        | estimated_cost | estimated_rows | cascade_tier
-- Aggregate { ... }| 142.0          | 12             | Aggregate
```

Implemented in [`crates/joule-db-query/src/planner.rs`](../../crates/joule-db-query/src/planner.rs) (`CascadeTier`, `PlanNode::cascade_tier()`) and surfaced by `explain_plan()` in [`execution.rs`](../../crates/joule-db-query/src/execution.rs). Closes Open Item §10.5 in [`MGAI-SPEC-DOMAIN-JOULEDB.md`](../MGAI-SPEC-DOMAIN-JOULEDB.md) (landed 2026-05-19).

## 6. Vector ops

```sql
-- k-NN against a 768-d embedding
SELECT title, embedding <-> '[0.12, 0.45, ...]'::vector AS distance
FROM articles
ORDER BY distance
LIMIT 10;

-- Hypervector similarity (HDC, returns µJ-class energy)
SELECT id, hv_similarity(hv_concept, hv_query) AS sim
FROM concepts
WHERE hv_similarity(hv_concept, hv_query) > 0.6;
```

Operators: `<->` (L2), `<#>` (cosine), `<=>` (dot product). HDC functions: `hv_bind`, `hv_bundle`, `hv_permute`, `hv_similarity`, `hv_random`. See [`MGAI-HDC-REFERENCE.md`](../MGAI-HDC-REFERENCE.md) for the primitives.

## 7. Time-travel (system-versioned reads)

Engine-level time-travel exists; the SQL surface is partial — currently you can read at a transaction id via `AS OF SYSTEM TIME` but the full temporal-predicate surface (`FOR SYSTEM_TIME BETWEEN`, etc.) is not yet exposed. Open Item §10.2 in the domain spec.

```sql
SELECT * FROM accounts AS OF SYSTEM TIME '2026-05-01 00:00:00';
```

## 8. Where it differs from PostgreSQL

| Behaviour | PostgreSQL | JouleDB |
|---|---|---|
| Unknown column reference | Could silently return NULL | **Errors** with `COLUMN_NOT_FOUND` via `validate_column_refs()` — fixed in May 2026 |
| `EXPLAIN` output | plan tree + cost | plan tree + cost + **`cascade_tier`** column (energy class) |
| Energy receipt | n/a | Every query returns `total_cost_uwh` in the `Done` frame |
| `LATERAL` joins | yes | partial — basic forms work, complex correlations untested |
| Recursive CTEs (`WITH RECURSIVE`) | yes | yes (see also [`QUERY-DATALOG.md`](QUERY-DATALOG.md) for native recursion) |
| Stored procedures (`CREATE FUNCTION ... LANGUAGE …`) | yes | `LANGUAGE wasm` only (via `joule-db-server::wasm-triggers`) |
| `LISTEN` / `NOTIFY` | yes | replaced by JWP subscriptions ([`subscriptions.rs`](../../crates/joule-db-server/src/subscriptions.rs)) |

## 9. Examples

```sql
-- Window function with grouped aggregate
SELECT
    department,
    employee_name,
    salary,
    AVG(salary) OVER (PARTITION BY department) AS dept_avg,
    salary - AVG(salary) OVER (PARTITION BY department) AS delta_from_avg
FROM employees
ORDER BY department, salary DESC;

-- Vector k-NN with energy budget
EXPLAIN ANALYZE
SELECT title, body, embedding <#> '[0.1, 0.2, ...]'::vector AS dist
FROM articles
WHERE language = 'en'
ORDER BY dist
LIMIT 5;

-- Recursive CTE: reachability
WITH RECURSIVE reach AS (
    SELECT src, dst FROM edges
    UNION ALL
    SELECT r.src, e.dst FROM reach r JOIN edges e ON r.dst = e.src
)
SELECT DISTINCT dst FROM reach WHERE src = 'a';
```

## 10. See also

- [`crates/joule-db-query/README.md`](../../crates/joule-db-query/README.md) — engine overview + invariant fuzz status
- [`MGAI-JWP-PROTOCOL.md`](../MGAI-JWP-PROTOCOL.md) — JWP frames that carry SQL queries
- [`docs/jouledb/SDK-RUST.md`](SDK-RUST.md) — Rust client SDK
- [`docs/jouledb/SDK-ODBC.md`](SDK-ODBC.md) — Excel / Tableau / Power BI via ODBC

---

*Drafted 2026-05-18 as wave 3 of the JouleDB documentation parity pass.*
