# JouleDB Gremlin Reference

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-db-query/src/gremlin.rs`](../../crates/joule-db-query/src/gremlin.rs) (663 LOC)
**Executor:** [`crates/joule-db-server/src/gremlin_executor.rs`](../../crates/joule-db-server/src/gremlin_executor.rs)

Gremlin is the graph traversal language of Apache TinkerPop. JouleDB supports the core traversal steps for graph navigation against the same shared storage that backs SQL/Cypher/SPARQL.

## 1. Supported syntax

```text
g.V().has("name", "Alice").out("knows").values("name")
g.V(1).outE("created").inV().path()
g.V().hasLabel("person").count()
g.addV("person").property("name", "Dave")
```

## 2. Supported steps

| Category | Steps |
|---|---|
| **Vertex/edge access** | `V()`, `V(id)`, `E()`, `E(id)` |
| **Filter** | `has`, `hasLabel`, `hasNot`, `hasKey`, `hasValue`, `where`, `is` |
| **Traversal** | `out`, `outE`, `outV`, `in`, `inE`, `inV`, `both`, `bothE`, `bothV`, `to`, `from` |
| **Modulation** | `as`, `select`, `by`, `order`, `limit`, `range`, `tail`, `skip`, `dedup` |
| **Projection** | `values`, `valueMap`, `properties`, `keys`, `label`, `id`, `path`, `project` |
| **Aggregation** | `count`, `sum`, `mean`, `min`, `max`, `fold`, `unfold`, `group`, `groupCount` |
| **Mutation** | `addV`, `addE`, `property`, `drop` |
| **Branch** | `union`, `choose`, `coalesce`, `optional`, `repeat`, `until`, `times` |
| **Sideeffects** | `aggregate`, `store`, `sack` |

## 3. Examples

```groovy
// People Alice knows, ordered by age
g.V().has("name", "Alice").out("knows")
  .order().by("age", asc)
  .values("name")

// Friends of friends, excluding direct friends
g.V().has("name", "Alice").out("knows").as("friend")
  .out("knows").where(neq("friend"))
  .where(not(__.in("knows").has("name", "Alice")))
  .dedup()
  .values("name")

// Add a vertex and an edge
g.addV("person").property("name", "Dave").as("dave")
  .addE("knows").from("dave").to(g.V().has("name", "Alice"))

// Repeat traversal: friends-of-friends-of-friends
g.V().has("name", "Alice")
  .repeat(out("knows")).times(3)
  .dedup()
```

## 4. Where it differs from TinkerPop reference

- **Same substrate as SQL** — vertices and edges are rows in JouleDB tables.
- **Energy receipts** — every traversal returns total µWh cost via the JWP `Done` frame.
- **GraphSON / Gryo serialization** — supported on the HTTP surface.
- **TinkerPop server protocol** — partial; the canonical wire is JWP.

## 5. See also

- [`crates/joule-db-query/README.md`](../../crates/joule-db-query/README.md)
- [`QUERY-CYPHER.md`](QUERY-CYPHER.md) — declarative graph queries
- [`QUERY-SPARQL.md`](QUERY-SPARQL.md) — RDF triple-pattern queries
