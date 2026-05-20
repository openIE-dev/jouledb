# JouleDB Cypher Reference

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-db-query/src/cypher.rs`](../../crates/joule-db-query/src/cypher.rs) (1,433 LOC)
**Executor:** [`crates/joule-db-server/src/cypher_executor.rs`](../../crates/joule-db-server/src/cypher_executor.rs)
**Sister docs:** [`QUERY-SQL.md`](QUERY-SQL.md), [`QUERY-GREMLIN.md`](QUERY-GREMLIN.md), [`QUERY-SPARQL.md`](QUERY-SPARQL.md), [`QUERY-DATALOG.md`](QUERY-DATALOG.md)

JouleDB Cypher is Neo4j-compatible. Same parser dialect, executes against the same shared storage substrate as SQL — so a graph query and a SQL query can hit the same underlying table.

## 1. Supported clauses

| Clause | Status |
|---|---|
| `MATCH` (with optional `OPTIONAL MATCH`) | yes |
| `WHERE` | yes |
| `RETURN` | yes |
| `WITH` (chained pipelines) | yes |
| `CREATE`, `MERGE`, `DELETE`, `DETACH DELETE` | yes |
| `SET`, `REMOVE` | yes |
| `UNWIND` (with list literals, not just identifiers) | yes |
| `ORDER BY`, `LIMIT`, `SKIP` | yes (`ORDER BY` correctly matches by name, not enumerate index) |
| `CALL` (procedures) | partial |
| `LOAD CSV` | partial |
| `FOREACH` | yes |

## 2. Operators and predicates

| Surface | Status |
|---|---|
| `=`, `<>`, `<`, `<=`, `>`, `>=` | yes |
| `AND`, `OR`, `NOT`, `XOR` | yes |
| `IN [list]` and `NOT IN [list]` | yes |
| `IS NULL` / `IS NOT NULL` | yes |
| `CONTAINS`, `STARTS WITH`, `ENDS WITH` | yes (with correct `Token::With` handling) |
| `=~` (regex) | yes |
| Subtraction (`Token::Dash` alias) | yes |
| `count()`, `sum()`, `avg()`, `min()`, `max()`, `collect()` | yes |
| Grouped aggregation in `RETURN`/`WITH` (implicit group by non-aggregate items) | yes |
| Path patterns: `()-[]-()`, variable-length `*1..3`, named relationships | yes |
| Time-travel: `AS OF <ts>` and `FOR SYSTEM_TIME AS OF <ts>` | yes (landed 2026-05-19) |

## 2a. Time-travel (`AS OF`)

System-versioned tables can be queried at a historical instant directly in Cypher — no manual `valid_from`/`valid_to` predicate plumbing:

```cypher
// Accounts as they stood on Jan 1 2026
MATCH (a:Account)
AS OF datetime('2026-01-01')
RETURN a.id, a.balance;

// SQL:2011-style spelling, identical semantics
MATCH (a:Account)
FOR SYSTEM_TIME AS OF datetime('2026-01-01')
WHERE a.region = 'us-east'
RETURN a;
```

`AS OF <ts>` is **desugared transparently** by the parser into the conjunctive predicate `valid_from <= <ts> AND <ts> < valid_to`, folded into an existing `WHERE` (or inserted as a synthetic `WHERE` right after the `MATCH`). The executor only ever sees the property predicates it already evaluates — so the temporal pin is enforced with zero executor changes and the same cost characteristics as the hand-written form. Multiple `AS OF` clauses conjoin. Both spellings (`AS OF`, `FOR SYSTEM_TIME AS OF`) are equivalent. See [`QUERY-TIME-TRAVEL.md`](QUERY-TIME-TRAVEL.md) for the validity model.

## 3. Examples

```cypher
// Find friends-of-friends
MATCH (me:Person {name: 'Alice'})-[:KNOWS*2..3]-(fof:Person)
WHERE fof.name <> 'Alice'
RETURN DISTINCT fof.name, count(*) AS shared_paths
ORDER BY shared_paths DESC
LIMIT 10;

// UNWIND with list literal
UNWIND [1, 2, 3, 4, 5] AS x
MATCH (n:Number {value: x})
RETURN n;

// Aggregation with implicit grouping
MATCH (e:Employee)-[:WORKS_IN]->(d:Department)
RETURN d.name, count(e) AS headcount, avg(e.salary) AS avg_salary
ORDER BY headcount DESC;
```

## 4. Where it differs from Neo4j

- **Same storage as SQL** — no separate graph engine. Vertices and edges are rows; the executor uses [`graph.rs`](../../crates/joule-db-query/src/graph.rs) operators.
- **Energy receipts** — every query returns joule cost via the JWP `Done` frame.
- **APOC procedures** — not implemented. Use SQL functions or wasm-trigger procedures instead.
- **GDS (Graph Data Science) library** — not implemented. The [`mgai-vla`](../../crates/mgai-vla/) and [`joule-db-features`](../../crates/joule-db-features/) graph operators are the equivalent.

## 5. See also

- [`crates/joule-db-query/README.md`](../../crates/joule-db-query/README.md)
- [`QUERY-SPARQL.md`](QUERY-SPARQL.md) — RDF-style graph queries on the same substrate
- [`QUERY-GREMLIN.md`](QUERY-GREMLIN.md) — graph traversal DSL alternative
