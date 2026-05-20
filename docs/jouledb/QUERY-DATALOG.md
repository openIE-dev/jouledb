# JouleDB Datalog Reference

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-db-query/src/datalog.rs`](../../crates/joule-db-query/src/datalog.rs) (1,320 LOC)
**Executor:** [`crates/joule-db-server/src/datalog_executor.rs`](../../crates/joule-db-server/src/datalog_executor.rs)

Datalog enables recursive queries — transitive closure, reachability, rule-based inference — that no other JouleDB query projection supports natively.

## 1. Syntax

```text
% Rules (head :- body)
reachable(X, Y) :- edge(X, Y).
reachable(X, Y) :- edge(X, Z), reachable(Z, Y).

% Queries (prefix with ?-)
?- reachable("a", X).

% Facts (head with no body)
edge("a", "b").
edge("b", "c").

% Negation (stratified)
not_reachable(X, Y) :- node(X), node(Y), not reachable(X, Y).

% Aggregation
count_paths(X, count(Y)) :- reachable(X, Y).
```

## 2. Supported

| Surface | Status |
|---|---|
| Facts (ground atoms) | yes |
| Rules with conjunction | yes |
| Recursive rules | yes |
| Stratified negation (`not p(X)`) | yes |
| Aggregation in rule heads (`count`, `sum`, `min`, `max`, `avg`) | yes |
| Query directive (`?- goal.`) | yes |
| Multi-rule predicates (disjunction via multiple rules) | yes |
| Anonymous variables (`_`) | yes |
| Comparison built-ins (`<`, `>`, `=`, `!=`) | yes |
| Arithmetic built-ins | yes |
| String built-ins | partial |
| Magic-set transformation | yes |
| Semi-naïve evaluation | yes |

## 3. Where it differs from Prolog

- **Pure Datalog** — no cuts, no side effects, terminates on stratified programs
- **Closed-world assumption** — `not p(X)` is "X is not known to satisfy p"
- **Bottom-up evaluation** by default; top-down magic-set transform for goal-directed queries
- **Stored on the same substrate** as SQL/Cypher tables — `edge(X, Y)` reads from the `edge` table

## 4. Examples

```datalog
% Ancestor reachability
parent("Alice", "Bob").
parent("Bob", "Carol").
parent("Carol", "Dave").

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

?- ancestor("Alice", X).
% Alice → Bob, Carol, Dave

% Stratified negation
node("a"). node("b"). node("c"). node("d").
edge("a", "b"). edge("b", "c").
disconnected(X, Y) :- node(X), node(Y), X != Y, not reachable(X, Y).
reachable(X, Y) :- edge(X, Y).
reachable(X, Y) :- edge(X, Z), reachable(Z, Y).
?- disconnected("a", X).
% a → d  (a can reach b and c but not d)
```

## 5. Performance

Semi-naïve evaluation only computes the delta at each iteration. For dense graphs the magic-set transform reduces work significantly. WCOJ ([`wcoj.rs`](../../crates/joule-db-query/src/wcoj.rs)) handles N-way join patterns in rule bodies without materialised intermediates.

## 6. See also

- [`crates/joule-db-query/README.md`](../../crates/joule-db-query/README.md)
- [`QUERY-SQL.md`](QUERY-SQL.md) §"Recursive CTEs" — the SQL-flavoured equivalent
- [`QUERY-SPARQL.md`](QUERY-SPARQL.md) — recursive triple-pattern queries
