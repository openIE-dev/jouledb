# joule-db-query

Query language parsers and execution engine for JouleDB.

`joule-db-query` is the polyglot query layer — seven query languages compile down to one shared execution plan over [`joule-db-core`](../joule-db-core/). All seven are default-on; pick a subset via cargo features if you want a smaller build.

## Supported languages (all default-on)

| Language | Module | Source style |
|---|---|---|
| **SQL** | [`sql.rs`](src/sql.rs) | ANSI subset + window functions, CTEs, set operations, JSON ops, `EXPLAIN` |
| **Cypher** | [`cypher.rs`](src/cypher.rs) | Neo4j-compatible graph queries; `MATCH` / `CREATE` / `RETURN` / `WITH` / `UNWIND` / `MERGE` |
| **CQL** | [`cql.rs`](src/cql.rs) | Cassandra Query Language — wide-column DSL |
| **GraphQL** | [`graphql.rs`](src/graphql.rs) | Standard GraphQL for API-style fetches |
| **Datalog** | [`datalog.rs`](src/datalog.rs) | Logic-rule queries with recursion |
| **SPARQL** | [`sparql.rs`](src/sparql.rs) | RDF triple-store queries (SPARQL 1.1) |
| **Gremlin** | [`gremlin.rs`](src/gremlin.rs) | Apache TinkerPop graph traversals |
| **InfluxQL / PromQL** | [`timeseries.rs`](src/timeseries.rs) | Time-series queries |

Per-language behaviour and currently-supported subsets are documented in `docs/jouledb/QUERY-*.md`.

## Execution engine

| Module | Role |
|---|---|
| [`execution.rs`](src/execution.rs) | Plan dispatcher — joins, aggregation, window functions, ORDER BY, LIMIT |
| [`planner.rs`](src/planner.rs) | Cost-based optimiser, plan cache, index advisor |
| [`adaptive/`](src/adaptive/) | Feedback-driven plan adaptation |
| [`wcoj.rs`](src/wcoj.rs) | Worst-case-optimal join (leapfrog triejoin) for N-way joins without materialised intermediates |
| [`wcoj_cost.rs`](src/wcoj_cost.rs) | Online learned cost model for WCOJ variable ordering — cardinality stats + RLS linear model + atom-coverage features; deviates from the frequency heuristic only on a predicted strict improvement (never-worse-than-baseline floor). Trains on deterministic leapfrog work via `execute_wcoj_learning()`. |
| [`columnar.rs`](src/columnar.rs) | Arrow `RecordBatch` push-down for SIMD aggregates |
| [`amorphic_executor.rs`](src/amorphic_executor.rs) | JIT struct-materialisation via [`joule-db-amorphic`](../joule-db-amorphic/) HDC semantic encoding |
| [`vector.rs`](src/vector.rs) | k-NN + similarity search on hypervector columns |
| [`functions.rs`](src/functions.rs) | Built-in functions: LIKE, regex, substring, tf-idf, JSON, string, math |
| [`storage_executor.rs`](src/storage_executor.rs) | Direct B-tree storage executor (no plan tree) |
| [`fuzz/`](./fuzz/) | Corpus-driven invariant fuzzer — INV-1 through INV-2422 |

## Feature flags

| Feature | Default | Purpose |
|---|---|---|
| `sql` | yes | SQL parser + executor |
| `cypher` | yes | Cypher parser + executor |
| `cql` | yes | CQL parser + executor |
| `graphql` | yes | GraphQL parser + executor |
| `datalog` | yes | Datalog parser + executor |
| `sparql` | yes | SPARQL parser + executor |
| `gremlin` | yes | Gremlin parser + executor |
| `timeseries` | yes | InfluxQL + PromQL |
| `storage-executor` | yes | Direct storage executor for B-tree scans (requires `sql`) |
| `parallel` | yes | Rayon-backed parallel execution |
| `arrow-execution` | off | Arrow columnar execution path |
| `datafusion-backend` | off | DataFusion as the columnar backend |
| `amorphic` | off | Amorphic executor (requires `sql`) |
| `hdc` | off | HDC vector ops via [`joule-db-hdc`](../joule-db-hdc/) + [`joule-db-domains`](../joule-db-domains/) |
| `analytical` | off | Analytical extensions |

## Usage

```rust,ignore
use joule_db_query::{QueryEngine, sql::SqlParser};

let parser = SqlParser::new();
let query = parser.parse("SELECT * FROM users WHERE id = 1")?;
```

## Tests

551 `#[test]` / `#[tokio::test]` annotations in `src/`. Fuzz harness in [`fuzz/`](./fuzz/) — every parser gets corpus-driven mutation testing.

## Recent fixes (May 2026)

Every limitation called out in prior audits is now closed in code:

- NTILE / FIRST_VALUE / LAST_VALUE / NTH_VALUE / PERCENT_RANK / CUME_DIST window functions
- LAG / LEAD default values (including negative literals via `Unary { Neg, Literal }`)
- SUM(DISTINCT) / AVG(DISTINCT) — aggregate-with-DISTINCT now dispatches correctly
- Multi-table JOINs — qualified-name fallback bug in `find_column_index()` fixed
- Silent NULL on unknown columns → now `COLUMN_NOT_FOUND` error via `validate_column_refs()`
- CQL `>`, `<`, `>=`, `<=`, `!=` comparison operators
- CQL `COUNT`/`SUM`/`AVG`/`MIN`/`MAX` aggregates
- Cypher `ORDER BY` now matches by column name (was sorting by enumerate position)
- Cypher `UNWIND` list literals
- Cypher `CONTAINS` / `STARTS WITH` / `ENDS WITH` (Token::With handling)
- Cypher subtraction via `Token::Dash` alias
- Cypher grouped aggregation in `RETURN` / `WITH` — implicit group keys from non-aggregate items
- Cypher `IN [list]` / `NOT IN [list]`
- Cypher `IS NULL` / `IS NOT NULL`
- Date tokens (year/month/day/hour/minute/second) as column names
- CTE scope in `UNION` / `EXCEPT` / `INTERSECT` RHS
- Table-level `PRIMARY KEY (col1, col2)` constraint
- `GROUP BY` ordinal references (`GROUP BY 1, 2`)
- Nested window functions in expressions (`score - AVG(score) OVER ()`)
- `pg_catalog` virtual tables (`pg_type`, `pg_namespace`, `pg_class`, `pg_attribute`, `pg_database`, `pg_settings`)

## See also

- [joule-db-server](../joule-db-server/) — wire-protocol dispatch (one executor module per language)
- [WHITEPAPER-JOULEDB-2026-04.md](../../docs/WHITEPAPER-JOULEDB-2026-04.md)
- `docs/jouledb/QUERY-*.md` — per-language references *(in progress)*
