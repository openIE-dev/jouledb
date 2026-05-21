# JouleDB SPARQL Reference

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-db-query/src/sparql.rs`](../../crates/joule-db-query/src/sparql.rs) (1,019 LOC)
**Executor:** [`crates/joule-db-server/src/sparql_executor.rs`](../../crates/joule-db-server/src/sparql_executor.rs)

SPARQL 1.1 over JouleDB. Triple patterns map directly to the amorphic HDC knowledge core, which means SPARQL queries can use **approximate-match** semantics — useful for fuzzy knowledge-graph traversal alongside the exact-match path.

## 1. Supported

| Surface | Status |
|---|---|
| `SELECT` / `CONSTRUCT` / `ASK` / `DESCRIBE` | yes |
| `WHERE` with triple patterns | yes |
| `OPTIONAL { … }` | yes |
| `UNION` | yes |
| `FILTER (expr)` | yes |
| `BIND (expr AS ?var)` | yes |
| `GROUP BY`, `HAVING` | yes |
| `ORDER BY`, `LIMIT`, `OFFSET` | yes |
| `PREFIX` declarations | yes |
| Property paths (`*`, `+`, `?`, `/`, `|`, `^`) | yes |
| Sub-queries | yes |
| Aggregates (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `GROUP_CONCAT`, `SAMPLE`) | yes |
| Built-in functions (`STR`, `LANG`, `DATATYPE`, `BOUND`, `IRI`, etc.) | yes |
| Update operations (`INSERT DATA`, `DELETE DATA`, `MODIFY`) | yes |
| Federated queries (`SERVICE`) | partial |
| Approximate triple matching (HDC-backed) | **JouleDB-specific** |

## 2. Examples

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?email
WHERE {
  ?person foaf:name ?name .
  ?person foaf:mbox ?email .
  OPTIONAL { ?person foaf:age ?age }
  FILTER (?age > 18)
}
ORDER BY ?name
LIMIT 10
```

```sparql
# Property path: ancestors via parent*
PREFIX : <http://example/>
SELECT ?ancestor WHERE { :alice :parent* ?ancestor }
```

```sparql
# Approximate match (JouleDB-specific) — finds triples
# semantically similar to the pattern even without exact term match
PREFIX : <http://example/>
SELECT ?animal WHERE {
  ?animal :similarTo :dog .   # HDC similarity, not literal equality
}
```

## 3. Approximate matching via HDC

SPARQL triple patterns can target either exact triples or HDC-encoded triples. When a triple's subject/predicate/object is stored as a hypervector (via [`joule-db-amorphic`](../../crates/joule-db-amorphic/) or [`joule-db-domains`](../../crates/joule-db-domains/)), the pattern matches by similarity rather than equality.

This is what powers the WN18RR benchmark result documented in [`WHITEPAPER-JOULEDB-2026-05.md`](../WHITEPAPER-JOULEDB-2026-05.md) §3 — 83.8% Hits@10 from approximate triple pattern matching at 9 µJ per prediction.

## 4. See also

- [`crates/joule-db-query/README.md`](../../crates/joule-db-query/README.md)
- [`MGAI-HDC-REFERENCE.md`](../MGAI-HDC-REFERENCE.md) — the HDC primitives that power approximate matching
- [`QUERY-CYPHER.md`](QUERY-CYPHER.md), [`QUERY-GREMLIN.md`](QUERY-GREMLIN.md), [`QUERY-DATALOG.md`](QUERY-DATALOG.md) — sibling graph projections
