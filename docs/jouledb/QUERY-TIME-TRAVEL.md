# JouleDB Time-Travel Query Reference

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-db-core/src/temporal/mod.rs`](../../crates/joule-db-core/src/temporal/mod.rs)
**Sister docs:** [`QUERY-SQL.md`](QUERY-SQL.md), [`QUERY-CYPHER.md`](QUERY-CYPHER.md)

JouleDB supports **system-versioned temporal tables** — every row carries `(valid_from, valid_to)` timestamps, and queries can read the database as it was at any historical point. Historical rows are immutable; only the latest version is mutable.

This was Open Item §10.2 in the domain spec ([`MGAI-SPEC-DOMAIN-JOULEDB.md`](../MGAI-SPEC-DOMAIN-JOULEDB.md)): the engine has full support; the SQL/Cypher front-end exposes a partial surface today.

---

## 1. The model

Each row in a temporal table stores a `Validity` interval:

| Field | Type | Meaning |
|---|---|---|
| `valid_from` | `i64` (µs since epoch) | When this version became current (inclusive) |
| `valid_to` | `i64` (µs since epoch, or `i64::MAX`) | When this version stopped being current (exclusive). `MAX` means still current. |

Operations:

| SQL | Effect on `valid_from` / `valid_to` |
|---|---|
| `INSERT` | New row: `(now, MAX)` |
| `UPDATE` | Close old: `(old.from, now)`. New: `(now, MAX)`. |
| `DELETE` | Close: `(row.from, now)`. Row not deleted from disk. |
| `SELECT` | By default reads `valid_to = MAX` rows |
| `SELECT … AS OF ts` | Reads rows where `valid_from <= ts < valid_to` |
| `SELECT … FOR SYSTEM_TIME BETWEEN a AND b` | Reads rows whose validity overlaps `[a, b)` |
| `SELECT … FOR SYSTEM_TIME ALL` | Reads every version |

---

## 2. SQL surface — what's exposed today

| Surface | Status |
|---|---|
| `AS OF '<timestamp>'` | yes |
| `AS OF SYSTEM TIME '<timestamp>'` | yes |
| `FOR SYSTEM_TIME AS OF '<timestamp>'` | yes |
| `FOR SYSTEM_TIME BETWEEN '<a>' AND '<b>'` | yes |
| `FOR SYSTEM_TIME ALL` | yes |
| `FOR SYSTEM_TIME FROM '<a>' TO '<b>'` (SQL:2011 alternate spelling) | partial |
| `WITH SYSTEM VERSIONING` on `CREATE TABLE` | yes |
| `DROP SYSTEM VERSIONING` | partial |
| History table separation (`SYSTEM VERSIONING WITH HISTORY TABLE …`) | not implemented — versions live in the same table |

---

## 3. Examples

### 3.1 Create a temporal table

```sql
CREATE TABLE accounts (
    id BIGINT PRIMARY KEY,
    balance NUMERIC NOT NULL,
    valid_from TIMESTAMP NOT NULL,
    valid_to TIMESTAMP NOT NULL
) WITH SYSTEM VERSIONING;
```

### 3.2 Point-in-time read

```sql
-- Balances as they were on Jan 1, 2026
SELECT id, balance
FROM accounts
AS OF '2026-01-01T00:00:00Z';
```

### 3.3 Range read

```sql
-- All distinct balances that existed for account 42 during Q1
SELECT balance, valid_from, valid_to
FROM accounts
FOR SYSTEM_TIME BETWEEN '2026-01-01' AND '2026-04-01'
WHERE id = 42
ORDER BY valid_from;
```

### 3.4 Full history

```sql
SELECT *
FROM accounts
FOR SYSTEM_TIME ALL
WHERE id = 42
ORDER BY valid_from;
```

---

## 4. Cypher surface

Cypher exposes a dedicated time-travel clause (landed 2026-05-19) — no manual validity-column plumbing required:

```cypher
MATCH (a:Account)
AS OF datetime('2026-01-01')
WHERE a.id = 42
RETURN a.balance;

// SQL:2011-style spelling, identical:
MATCH (a:Account)
FOR SYSTEM_TIME AS OF datetime('2026-01-01')
WHERE a.id = 42
RETURN a.balance;
```

The parser desugars `AS OF <ts>` into the equivalent predicate `valid_from <= <ts> AND <ts> < valid_to`, folded into the query's `WHERE` (an existing one is AND-extended; otherwise a synthetic `WHERE` is inserted right after the `MATCH`). The executor never sees a special clause — the temporal pin reduces to the property predicates it already evaluates, so behaviour and cost match the hand-written form exactly. The earlier hand-written form remains valid:

```cypher
MATCH (a:Account)
WHERE a.valid_from <= datetime('2026-01-01')
  AND datetime('2026-01-01') < a.valid_to
  AND a.id = 42
RETURN a.balance;
```

Implemented in [`crates/joule-db-query/src/cypher.rs`](../../crates/joule-db-query/src/cypher.rs) (`CypherClause::AsOf`, `desugar_temporal`). Closes the remaining Cypher half of Open Item §10.2.

---

## 5. Energy / storage cost

Time-travel is not free. Cost properties:

- **Write amplification:** every UPDATE writes two rows (close old + insert new). DELETE is one write (close).
- **Index size:** indexes on temporal tables include `(valid_from, valid_to)` to make AS-OF queries log-time. Roughly 2× index size.
- **Read cost for historical queries:** O(log N + range_size) where N is the table version count.
- **Vacuum:** historical rows are append-only. There is no "garbage collect old versions" — that's a design choice (audit trail). If you need to drop history, copy current rows to a fresh non-versioned table.

Energy receipt for an `AS OF` query is typically ~2× a same-shape non-temporal query (the index scan walks one extra dimension).

---

## 6. Use cases this enables

- **Audit trails** — "what did the database look like at the time of the incident?"
- **Slowly Changing Dimensions** (SCD type 2) — no separate history table; the main table is the history.
- **Reproducible analytics** — "what would yesterday's report have said?"
- **Schema migration verification** — "did the new code see the same data the old code did?"
- **Compliance** — regulatory requirements for tamper-evident history.

---

## 7. Limitations

- **Versions live in-table.** No separate history table per SQL:2011 — versions and current rows coexist. Indexes include validity columns.
- ~~**Cypher `AS OF` clause not yet exposed.**~~ **Closed 2026-05-19** — `AS OF` and `FOR SYSTEM_TIME AS OF` are first-class Cypher clauses, desugared transparently to validity predicates (§4).
- **No automatic version pruning.** History is intentionally permanent; if you need to drop old versions, copy to a fresh non-versioned table.
- **Bi-temporal support partial.** Application-time periods (separate from system-time periods) are not modelled.

---

## 8. See also

- [`crates/joule-db-core/src/temporal/mod.rs`](../../crates/joule-db-core/src/temporal/mod.rs) — engine implementation
- [`QUERY-SQL.md`](QUERY-SQL.md) §7 — earlier mention in the SQL reference
- [`MGAI-SPEC-DOMAIN-JOULEDB.md`](../MGAI-SPEC-DOMAIN-JOULEDB.md) §10.2 — Open Item this doc closes
- SQL:2011 spec — system-versioned tables (the standard this surface tracks)
