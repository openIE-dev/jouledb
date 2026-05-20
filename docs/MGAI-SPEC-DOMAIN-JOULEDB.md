# MGAI Domain Specification: jouledb

**Version 1.0 — 2026-05-09**
**Domain:** the persistence layer of the OpenIE stack — ACID database + HDC runtime + knowledge graph + energy meter
**Crates:** 27 in the `joule-db` domain bucket per the crate inventory CSV
**Anchor docs:** `docs/WHITEPAPER-JOULEDB-2026-04.md` (preserved v0.1, 390 lines), `ARCHITECTURE.md`, `jouledb-sota-architecture.md`, `conformance-report.md`, `the-fix-1.md`, `the-fix-2.md`

---

## 1. Purpose

JouleDB is the OpenIE storage layer, with three integrated capabilities:

1. **A production ACID database** — durable, transactional, crash-recoverable
2. **A hyperdimensional computing runtime** — 15 irreducible primitives
3. **A knowledge system** — starts empty, learns from live interaction

Every operation is metered in joules. Every query language is a Translator on top of one shared substrate.

This document is a **thin wrapper** over the canonical anchor docs. The detailed architecture lives in those documents; this spec captures the audit-pass view: state, scope, integration with the rest of the OpenIE stack, and open items.

---

## 2. Crate manifest (27 crates)

Per `MGAI-CRATE-INVENTORY-2026-05-09.csv`, the `joule-db` domain bucket contains 27 crates totalling 575 .rs files (the densest non-substrate domain — average 21 .rs per crate). Major members:

| Crate | Role |
|---|---|
| `joule-db-core` | Core types — pages, B-trees, transactions, WAL |
| `joule-db-local` | Local storage backend |
| `joule-db-query` | Query engine + cross-language compiler |
| `joule-db-server` | Network server (pgwire compatible + native) |
| `joule-db-hdc` | Hyperdimensional computing primitives |
| `joule-db-recover` | Forensic recovery tool |
| `joule-db-tantivy` | Full-text index integration |
| `jouledb-ai-runtime` | AI-side cascade dispatch into the engine |

Plus subsystem-specific crates (storage backends, index types, query-language frontends, RPC).

---

## 3. Headline capability

### 3.1 Multi-language query surface (SOTA unification)

Per spec v1.5 §3 and the SOTA-unification work (April 2026, 124 tests, 9 new files):

| Query language | Status |
|---|---|
| SQL | Full surface — pgwire compatible, including window functions, CTEs (recursive), subqueries, set operations, GROUP BY ordinal, NTILE, FIRST/LAST/NTH_VALUE, LAG/LEAD with default, PERCENT_RANK, CUME_DIST |
| Cypher | RETURN/WITH grouped aggregation, ORDER BY by name, UNWIND list literals, IN [list], IS NULL / IS NOT NULL, CONTAINS / STARTS WITH / ENDS WITH, subtraction with `-` |
| CQL | Comparison operators, aggregate functions (COUNT/SUM/AVG/MIN/MAX) |
| Datalog | Recursive rules, stratified negation |
| SPARQL | Triple patterns, basic graph patterns |
| Gremlin | Step-based traversals |
| GraphQL | Schema-driven query |

5 index types, 18+ graph algorithms, WCOJ (Worst-Case Optimal Joins), time-travel queries.

### 3.2 Bug-density discipline

Per memory's bug-hunt history: 10+ sessions, 60+ bugs fixed across `joule-db-core`, `joule-db-local`, `joule-db-query`, `joule-db-server`. Major fix categories:

- B-tree overflow page chain (large values)
- WAL payload validation
- pgwire protocol hardening
- Query engine safety (negative `usize` casts, window frame bounds)
- Buffer pool free_page ordering
- Workflow energy overflow / time underflow
- Lock poisoning recovery
- Multi-table JOIN equi-join key extraction
- SUM(DISTINCT) / AVG(DISTINCT) handling
- FIRST_VALUE / LAST_VALUE / NTH_VALUE window functions
- Silent-NULL for non-existent column references (now errors)
- CQL parser comparison operators
- Cypher ORDER BY column matching
- LAG/LEAD negative default values
- Cypher RETURN/WITH grouped aggregation
- pg_catalog virtual tables

Plus 6,163+ server-lib tests and 180+ integration tests, all passing as of latest audit. The **invariant fuzzer** (`query.rs`) maintains 2,422 invariant tests (INV-1 through INV-2422), all passing.

### 3.3 Eliminated limitations (all 5 fixed)

| Original limitation | Resolution |
|---|---|
| NTILE window function | Implemented bucket assignment |
| UNION/EXCEPT/INTERSECT column count mismatch | Now returns error rather than silent truncation |
| Reserved words (threshold, key, similar, meaning, nearest, primary) | Allowed in identifiers and primary expressions |
| GROUP BY CASE expr | Verified working — test threshold was wrong |
| Cypher CREATE without variable | Verified working — tests had `CYPHER` prefix |

---

## 4. Energy receipt

Every JouleDB operation emits a per-call receipt:

```
{ "result": ..., "duration_secs": ..., "energy_joules": ..., "tier": "L0" | "L1" | "L2" | ... }
```

The tier annotation tells the caller which cascade level served the query. L0 = exact lookup; L1 = HDC similarity; L2+ = formula / solver dispatch.

This is the per-call equivalent of the `mantle-measure` per-op receipts. Aggregating receipts at the MCP layer (`ask-mcp::joule_web_compute`) produces session-level energy accounting.

---

## 5. JouleDB AI: the knowledge system

### 5.1 Layer 3 of the three-layer stack

Per WHITEPAPER-JOULEDB v0.1 §2.4: JouleDB AI is the knowledge system on top of the database (Layer 1) and HDC runtime (Layer 2). It starts empty and learns from live interaction.

### 5.2 Headline benchmark

On WN18RR knowledge graph: **83.8% Hits@10** (28 points above published SOTA at the time) at **9 microjoules per prediction** — approximately 10⁶× more energy-efficient than comparable GPU inference.

### 5.3 Tier-0 brain

Per `ARCHITECTURE.md`: a 3 KB tier-0 representation packs 34 patterns × 11 eigendims (3,264 bytes) plus 216 concepts and 1,253 edges. Energy cost: 0.1–1 µJ per query. This is the structural answer to "why does dense matmul win at recall?" — the tier-0 brain wins instead, at six orders of magnitude less energy.

---

## 6. Recovery and resilience

JouleDB has been through multiple production-recovery passes:

### 6.1 BufferPool::new_page bug (FIXED)

Per memory `project_scholar_jouledb_bug.md` (2026-04-21). Buffer pool was flushing empty+dirty placeholder pages under LRU eviction. Fix: seed `new_page` with a valid empty-leaf body. Backfill of 566K records re-ran clean.

### 6.2 save_metadata race (FIXED)

Per memory `project_scholar_jouledb_meta_race_bug.md` (2026-04-23). Fixed-path tmp file was being clobbered under concurrent `save_metadata` calls. Fix: `meta.wdb.tmp.<pid>.<seq>` + serialising mutex. Regression test: 400 concurrent calls.

### 6.3 Recovery 2026-05-02

Per memory `project_scholar_recovery_2026_05_02.md`. Forensic `joule-db-recover` tool plus B-tree rebuild from leaves (synthetic root 9107416, ~4M scholar_works leaves salvaged). Reads slow due to sparse fill; tantivy index degraded to 1.94M docs.

### 6.4 Re-ingest as repair playbook

Per memory `project_scholar_reingest_repair_2026_05_03.md`. Operational stance: JouleDB is treated as a cache of CC0/OA sources, not a system of record. Archive + re-ingest is the supported repair path. Cutover 2026-05-03 for the scholar deployment.

### 6.5 Iterator hardening 2026-05-07

Per memory `project_scholar_jouledb_iterator_hardening_2026_05_07.md`. 7 commits covering ingest 5–35× faster, row-count refresh 1800× faster, three iterator bugs fixed (depth-unbounded recursion, re-entry on error, silent count drop), `prefix_count` tree-walk + ancestor-stack cycle detect.

These are documented in `the-fix-1.md` and `the-fix-2.md` at the top level.

---

## 7. Conformance

`conformance-report.md` at the top level reports 18/18 fixtures passing. The conformance oracle is a Python-side reference implementation that the Rust port validates against (per `MGAI-CONFORMANCE-ORACLE.md`) — Tier 0 (deterministic bitwise) and Tier 4 (numerical with absolute tolerance 1e-3) covered.

---

## 8. Position in the OpenIE stack

```
                    [ user query / agent call ]
                                |
                                v
         ┌─────────────────────────────────────────┐
         │  ask-server / ask-mcp                    │  (cascade dispatch)
         └────────────────┬────────────────────────┘
                          │
                          ▼
         ┌─────────────────────────────────────────┐
         │  physical-cascade  / verity-cascade      │  (tier selection)
         └────────────────┬────────────────────────┘
                          │
                          ▼ L0 / L1 / L2
         ┌─────────────────────────────────────────┐
         │  JouleDB AI (Layer 3)                    │
         │  ─ knowledge graph                       │
         │  ─ tier-0 brain (3 KB, 1 µJ/query)       │
         └────────────────┬────────────────────────┘
                          │
                          ▼
         ┌─────────────────────────────────────────┐
         │  HDC Runtime (Layer 2)                   │
         │  ─ 15 irreducible primitives             │
         └────────────────┬────────────────────────┘
                          │
                          ▼
         ┌─────────────────────────────────────────┐
         │  JouleDB Database (Layer 1)              │
         │  ─ ACID, durable, crash-recoverable      │
         │  ─ 7 query languages                     │
         │  ─ 5 index types                         │
         │  ─ pgwire compatible                     │
         └─────────────────────────────────────────┘
```

JouleDB is **the persistence layer** referenced in `WHITEPAPER.md` v0.2 §1 (vertical-integration map). It sits below the cascade and serves L0–L1 tiers (and parts of L2 — closed-form lookups).

---

## 9. Code fruition

| Crate cluster | Tests | Stub markers | Fruition |
|---|---|---|---|
| `joule-db-core` | (counts in 6,163+ aggregate) | None flagged | Production |
| `joule-db-local` | (counts in 6,163+ aggregate) | None flagged | Production |
| `joule-db-query` | 6,163+ + 2,422 invariant fuzz | None flagged | Production |
| `joule-db-server` | 6,163+ aggregate + 180+ integration | None flagged | Production |
| `joule-db-hdc` | (subsystem tests) | None flagged | Production |
| `joule-db-recover` | (recovery-tool tests) | None flagged | Production |
| `jouledb-ai-runtime` | (tier-0 brain tests) | 1 cascade-stub | Active |

Per Phase 0a stub inventory: zero `todo!()` / `unimplemented!()` macros across the joule-db family. The bug-hunt discipline plus invariant fuzzer make this the most-tested subsystem in the workspace.

---

## 10. Open items

1. ~~**Tantivy index recovery from sparse-fill state** — per recovery 2026-05-02 the index degraded to 1.94M docs; rebuild flow exists but is operator-driven, not auto-recovered.~~ **Closed 2026-05-19** — `scholar-server` runs a background index-health monitor (`spawn_index_health_monitor` in [`crates/scholar-server/src/main.rs`](../crates/scholar-server/src/main.rs)) that detects sparse-fill via the existing row-count cache (no added I/O), transitions the index `Healthy → Degraded → Rebuilding → Healthy`, runs a throttled background rebuild, and hot-swaps the live handle without a process restart. Pure predicate `is_sparse_fill()` with 5 unit tests; 22/22 scholar-server tests pass. Operator runbook: [`RUNBOOK-RECOVERY.md`](jouledb/RUNBOOK-RECOVERY.md) §10.
2. ~~**Time-travel query language exposure**~~ **Closed 2026-05-19** — SQL `AS OF` / `FOR SYSTEM_TIME` and **Cypher** `AS OF` / `FOR SYSTEM_TIME AS OF` are both first-class. The Cypher clause desugars transparently to `valid_from <= ts AND ts < valid_to` ([`CypherClause::AsOf` + `desugar_temporal` in cypher.rs](../crates/joule-db-query/src/cypher.rs)), folding into an existing `WHERE` — zero executor changes. Docs: [`QUERY-TIME-TRAVEL.md`](jouledb/QUERY-TIME-TRAVEL.md), [`QUERY-CYPHER.md`](jouledb/QUERY-CYPHER.md) §2a. 512/512 query tests pass.
3. ~~**WCOJ optimiser cost model** — currently uses heuristic order; a learned model is queued.~~ **Closed 2026-05-19** — [`crates/joule-db-query/src/wcoj_cost.rs`](../crates/joule-db-query/src/wcoj_cost.rs) adds an online learned cost model: per-relation cardinality stats + a 12-feature map (incl. an order-sensitive atom-coverage prefix-cost signal + quadratic interactions) + a recursive-least-squares linear model (forgetting factor, deterministic at inference, weights persist as JSON). Selection enumerates candidate orders, scores each, and only deviates from the frequency baseline on a predicted strict improvement after ≥16 observations — a hard "never worse than baseline" floor. `execute_wcoj_learning()` trains it from real executions on **leapfrog work** (deterministic, the quantity ordering actually minimises). The frequency heuristic is retained as `frequency_order()` (the baseline/cold-start path; `compute_variable_order()` delegates to it). End-to-end validation in `wcoj.rs` proves correctness-invariance across orders, convergence to near-optimal, and no regression vs. the baseline. 508/508 query tests pass.
4. **HDC primitive count documentation** — ~~the README mentions "15 irreducible primitives" but the per-primitive doc lives inside the source~~. **Closed 2026-05-18** by [`MGAI-HDC-REFERENCE.md`](MGAI-HDC-REFERENCE.md).
5. ~~**JouleDB AI per-query cascade tier annotation** — currently in the energy receipt but not exposed via the SQL `EXPLAIN` surface.~~ **Closed 2026-05-19** — `EXPLAIN` now emits a `cascade_tier` column (`CascadeTier` enum + `PlanNode::cascade_tier()` in [`crates/joule-db-query/src/planner.rs`](../crates/joule-db-query/src/planner.rs)). Tier = max over the plan tree; one of Lookup / Formula / Extract / Aggregate / Reason. Documented in [`QUERY-SQL.md`](jouledb/QUERY-SQL.md) §5.

**Documentation tracking:** [`JOULEDB-DOCS-PUNCH-LIST.md`](JOULEDB-DOCS-PUNCH-LIST.md) is the master tracker for documentation parity work. Waves 1-3 (per-crate READMEs, whitepaper v0.2, wire / HDC / cloud / per-language / SDK / runbook / ACP coverage) landed 2026-05-18; wave 4 covers residual cross-links and the engineering-then-doc items above.

---

## 11. References

| Doc | Role |
|---|---|
| **[`WHITEPAPER-JOULEDB-2026-05.md`](WHITEPAPER-JOULEDB-2026-05.md)** | **v0.2 — current whitepaper** |
| `WHITEPAPER-JOULEDB-2026-04.md` | v0.1 — preserved as historical |
| `ARCHITECTURE.md` | Component-level diagram (workspace root) |
| `jouledb-sota-architecture.md` | SOTA-comparison + roadmap |
| `JOULEDB-DOCS-PUNCH-LIST.md` | Doc parity tracker |
| `MGAI-JWP-PROTOCOL.md` | JouleDB wire protocol spec |
| `MGAI-PGWIRE-COMPAT.md` | PostgreSQL wire compatibility |
| `MGAI-HDC-REFERENCE.md` | 15-primitive HDC reference |
| `jouledb/CLOUD-OPERATOR.md` | K8s operator runbook |
| `jouledb/CLOUD-API.md` | Cloud API narrative |
| `jouledb/CLOUD-BILLING.md` | Stripe billing pipeline |
| `jouledb/CLI-REFERENCE.md` | JouleDB CLI surface |
| `jouledb/MCP-JOULEDB.md` | JouleDB MCP bridge |
| `jouledb/QUERY-*.md` | Per-language references (9 docs) |
| `jouledb/SDK-*.md` | SDK quickstarts (5 docs) |
| `jouledb/RUNBOOK-*.md` | Operational runbooks (3 docs) |
| `jouledb/SOTA-SWOT-2026-05.md` | Database×AI landscape + honest SWOT (parallel-lane positioning) |
| `conformance-report.md` | 18/18 fixture status |
| `the-fix-1.md`, `the-fix-2.md` | Recovery / fix narratives |
| `MGAI-SPEC-DOMAIN-CASCADE.md` | Cascade tier dispatch (L0–L1 served by JouleDB) |
| `MGAI-SPEC-DOMAIN-MATHGROUND.md` | Algebra substrate |
| `MGAI-SPECIFICATION.md` v1.6 | Top-level MGAI spec |
| `WHITEPAPER.md` v0.2 §5 | OpenIE Stack Whitepaper — JouleDB chapter |

---

*Document opened 2026-05-09 by audit pass; references updated 2026-05-18 to reflect the wave-1/2/3 doc parity landings. Thin wrapper for the persistence layer; full architectural detail in the anchor docs.*
