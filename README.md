# JouleDB

**An energy-metered database engine. Every operation — store, query, HDC bind, cascade tier dispatch — is measured in joules, and the receipt is cryptographically anchorable.**

JouleDB takes one position: *the right unit of account for database + AI work is the verifiable joule, and a deterministic spine with the model at the leaves is how you minimize it.* It is not a competitor to a lakehouse or a warehouse — it is a different axis.

> Source-available under the **Business Source License 1.1** (converts to Apache-2.0 on the Change Date — see `LICENSE`). This is a **capability showcase**, published to demonstrate the architecture, not a managed service.

---

## What it is

One B-tree storage substrate carrying:

- **7 query languages** — SQL, Cypher, CQL, GraphQL, Datalog, SPARQL, Gremlin (+ InfluxQL/PromQL) — over one engine, not separate stores.
- **MVCC copy-on-write** with Phase 5/6 atomic publish, depth-bounded cycle-safe traversal, and a forensic page-walker recovery tool.
- **A deterministic cascade as the substrate**, not a router bolted in front of an LLM: `Lookup → Formula → Extract → Aggregate → Reason`. `EXPLAIN` reports the cascade tier per query.
- **An HDC runtime** — 15 irreducible primitives (bind / bundle / permute / Reflect / Inhibit / Spawn / …), zero-train one-shot learning. 83.8% Hits@10 on WN18RR at ~9 µJ/prediction, no training step.
- **Energy receipts** via `joule-db-energy`, Merkle-anchorable via `joule-db-ledger` — auditable without trusting the operator.
- **A learned WCOJ cost model** and **self-healing search index** (auto-recovery after a sparse-fill).
- Wire surface: a native binary protocol (JWP), PostgreSQL pgwire compatibility, WebSocket, HTTP, MCP.

## What it is not (read this)

This repository is published with its limitations stated, because a credible showcase states them:

- **Single-machine in production.** Raft / 2PC / sharding code ships and is tested; production deployments are single-instance. This is not a managed, multi-region service.
- **The server tier has unaudited `unwrap()`s** — a known panic surface.
- **Text generation is crude**; the zero-train knowledge result (WN18RR) is a knowledge-graph benchmark, not an agent-memory leaderboard score.
- **The AI bridge (`jouledb-ai-runtime`) is not in this repository.** It depends on a private AI library. JouleDB's HDC features here ride on the HDC *primitive math* (`inv-hdc-core`), not that library. The AI integration is described in the docs, not shipped.
- **The amorphic engine's pattern-language resolver is feature-gated off** in this build (it depends on a separate project). The storage-side amorphic path is intact; the pattern-lang-backed codegen path is not shipped.

For the full, unsugared assessment see `docs/WHITEPAPER-JOULEDB-2026-05.md` §5–6.

## Why it exists

In 2026 the database+AI field converged on one move — collapse OLTP/OLAP/AI into one governed substrate, put an agent on top — and re-derived two of JouleDB's design points independently: copy-on-write versioned storage with instant branching, and deterministic complexity-routing in front of the model. Those arrivals were measured in **dollars**. JouleDB was built on the same structure because the unit of account is the **verifiable joule**.

## Quickstart

```bash
cargo build --release -p joule-db-core
cargo test  -p joule-db-core --lib
```

- Rust client SDK — `docs/jouledb/SDK-RUST.md`
- C ABI / WASM / ODBC / LangGraph — `docs/jouledb/SDK-*.md`
- Per-query-language references — `docs/jouledb/QUERY-*.md`
- Wire protocol — `docs/MGAI-JWP-PROTOCOL.md`; pgwire compatibility — `docs/MGAI-PGWIRE-COMPAT.md`
- HDC primitive reference — `docs/MGAI-HDC-REFERENCE.md`
- Architecture & honest assessment — `docs/WHITEPAPER-JOULEDB-2026-05.md`

## Layout

`crates/joule-db-*` — the engine (core, local, query, server, client, c, browser, edge, gpu, branch, crdt, ternary, features, energy, viz, weights, ledger, langgraph, hdc, domains, amorphic, odbc, test-harness, benches, quickstart).
`crates/joule-cloud-*` — the managed-cluster operator surface (control plane, provisioner, API gateway, billing).
Supporting infra crates: `jwp` (wire protocol), `sigql`, `inv-mcp-core`, `jpb-core`, `inv-energy-core`, `inv-hdc-core`, and the `inv-auth`/post-quantum-crypto cluster.

## License

Business Source License 1.1 — see `LICENSE`. Source-available; non-production use granted; converts to Apache-2.0 on the Change Date. Not affiliated with, or endorsed by, any database vendor named in the documentation.
