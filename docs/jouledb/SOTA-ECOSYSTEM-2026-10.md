# JouleDB vs. SOTA: ecosystem and runtime gap map (Oct 10, 2026)

Baseline: openIE-dev/jouledb origin/main 6ac1712. Every JouleDB claim below points at code in that tree.
Vendor facts come from the URLs cited, accessed Oct 10, 2026. Nothing here proposes dropping or deferring an implemented capability.

## What JouleDB has today (from code)

| Axis | In code | Where |
|---|---|---|
| Wire: pgwire | Simple Q plus extended P/B/D/E/S/C/H/X, SCRAM. **No COPY** (no CopyIn/CopyOut messages) | crates/joule-db-server/src/pgwire.rs (3582 LOC), scram.rs |
| Wire: HTTP | REST `/api/v1/*` (ingest, export/{table}, query/similar, branches, backup, tenants, energy, memory), `/metrics`, `/health/*` | joule-db-server/src/lib.rs routes |
| Wire: other | JWP (own binary protocol, port+3), TCP binary, RESP, WebSocket, WebTransport, gRPC (tonic) | jwp_server.rs, tcp_server.rs, resp.rs, websocket.rs, webtransport.rs |
| MCP | JSON-RPC MCP transport + bridge | mcp_transport.rs (933 LOC), mcp_bridge.rs, inv-mcp-core |
| Arrow Flight SQL | **Absent**. `flight` hits are unrelated (in-flight counters) | — |
| Parquet | **Not real Parquet.** sigql/src/io/parquet.rs says it "uses a simplified binary format that mimics Parquet"; files are not readable by other tools | sigql/src/io/parquet.rs |
| Arrow | `arrow` 58 optional in joule-db-query (feature `arrow-execution`), internal only; no Arrow IPC import/export surface | joule-db-query/Cargo.toml |
| Iceberg/Delta attach | **Absent** (the `iceberg` hits are web asset ordering; `delta` hits are mutation deltas) | — |
| CSV/JSON | CLI import/export (csv, json, "parquet") | crates/bin/jouledb-cli/src/commands/data.rs |
| CDC/streaming | subscriptions, realtime, queue publish/subscribe/ack, mutation_delta, replication | subscriptions.rs, realtime/, replication.rs |
| SDKs | Rust only (joule-db-client: client, pool, protocol), C SDK doc. **No Python, JS/TS, Go, Java/JDBC packages** | crates/joule-db-client, docs/jouledb/SDK-C.md |
| Agent integrations | MCP server, LangGraph crate, agent memory API. **No LangChain/LlamaIndex packages, no published tool schemas** | joule-db-langgraph, agent_memory.rs |
| Vector | HNSW and others in VectorIndexManager, HDC; recall tests in-repo. **No published recall@k vs. ann-benchmarks** | vector_index.rs, joule-db-hdc |
| Observability | Prometheus `/metrics`, OTLP trace exporter config | metrics.rs, observability.rs |
| Energy receipts | Measured per-device joules (powermetrics, RAPL, NVML, EMI) on every routed op | power_meter.rs, nvml.rs, fabric.rs |
| Install | **Three conflicting stories**: (1) docs/src/install.md says `cargo binstall jouledb` / `cargo install jouledb`, but the crates.io `jouledb` package is a stub (stubs/jouledb, "Stub installer"); (2) release.yml checks out a *private* repo (openIE-dev/joulesperbit, needs a secret) and builds `crates/bin/jouledb-cli`, but names the binary `jouledb`. That binary is the client CLI, not the server; (3) README builds only the `joule-db-core` lib. No Dockerfile, no brew formula, no pip/npm | docs/src/install.md, stubs/jouledb, .github/workflows/release.yml, README.md |

## Competitor reference points (cited)

- DuckDB 1.5.0 (Mar 9, 2026): VARIANT, core GEOMETRY, new CLI; 1.5.2 brings DuckLake 1.0, Iceberg ALTER/updates, and a rebuilt WASM shell; 1.5.3 ships the Quack client-server protocol. https://duckdb.org/2026/03/09/announcing-duckdb-150 , https://duckdb.org/2026/04/13/announcing-duckdb-152 , https://duckdb.org/2026/05/20/announcing-duckdb-153
- MotherDuck is the hosted DuckDB 1.5 platform. https://motherduck.com/videos/whats-new-duckdb-15/
- LanceDB v0.40 (Oct 7, 2026): Python/Node/Rust/Java SDKs, OpenTelemetry metrics, table branches, torch dataloaders. https://github.com/lancedb/lancedb/releases/tag/python-v0.36.0 , https://releases.fru.dev/releases/lancedb-2026-10-07-lancedb-v0-40-0
- Databricks Lakebase: GA on AWS Feb 3, 2026 and on Azure Mar 3. It is serverless Postgres (Neon) with branching, scale-to-zero, PITR, and Lakebase Search. https://www.databricks.com/blog/databricks-lakebase-generally-available , https://docs.databricks.com/aws/en/release-notes/lakebase/ , https://neon.com/docs/introduction/neon-and-lakebase
- pgvector 0.8.x iterative scans; pgvectorscale 0.9 StreamingDiskANN with filtered recall. https://github.com/timescale/pgvectorscale/issues/87 , https://www.tencentcloud.com/document/product/409/79593
- Turso: Rust SQLite rewrite with MVCC (experimental), CDC, and Go/JS/Java/.NET/Python/Rust/WASM bindings; Postgres frontend underway. https://github.com/tursodatabase/turso , https://turso.tech/blog/a-new-modern-version-of-postgres-in-rust
- Qdrant Edge: embedded beta with Python and Rust bindings. https://qdrant.tech/documentation/edge/
- ClickHouse 26.x: full-text index GA, HNSW vector search, and leads ClickBench. https://clickhouse.com/blog/full-text-search-ga-release , https://clickhouse.com/resources/engineering/fastest-olap-databases
- Milvus 3.0 (Jul 2026): lake-native External Collections over Parquet/Lance/Iceberg. https://milvus.io/blog/announcing-milvus-3-lake-native-vector-search-and-a-more-powerful-retrieval-engine.md
- Weaviate 1.39 (Aug 2026): MMR, 4-bit RQ, gRPC-Web, HNSW snapshots. https://weaviate.io/blog/weaviate-1-39-release
- SurrealDB 3.3 (Sep 24, 2026): Postgres-client connectivity (`--postgres-bind`) and pre-filtered vector search. https://surrealdb.com/releases/3.3
- MongoDB 9.0 and Atlas Infinite (Sep 29, 2026). https://www.mongodb.com/company/newsroom/press-releases/mongodb-launches-mongodb-9-0-the-best-version-ever-built-and-atlas-infinite-for-ai-scale-demand
- Snowflake: Iceberg tables and Postgres integrations (via partners such as SurrealDB Sync). https://surrealdb.com/blog/announcing-snowflake-integration-for-surrealdb-from-data-warehouse-to-context-layer

Every vendor above ships a pip package, and most also ship npm, docker, and brew. None publishes joules per query.

## Ranked gaps (impact on adoption and SOTA claims vs. effort)

| # | Gap | Impact | Effort | Plan |
|---|---|---|---|---|
| 1 | One working install story (the three stories conflict; the release workflow depends on a private repo and ships the wrong binary) | Blocks everything | S | Release workflow builds the `jouledb` server from this repo; docker; brew formula; fix docs |
| 2 | Proven compatibility with real Postgres clients (psql, psycopg 3, SQLAlchemy, node-postgres, JDBC) in CI | Very high: unlocks every ORM and BI tool for free | M | CI job running real clients; fix what breaks |
| 3 | Python SDK with Arrow to pandas/polars plus the energy receipt | Very high (every competitor has one) | M | PyO3/maturin over HTTP/pgwire, Arrow |
| 4 | Real Parquet and Arrow IPC import/export (the current "parquet" is a lookalike) | High (interop claim is currently false) | M | `parquet`/`arrow` crates behind the existing io trait |
| 5 | Public benchmark harness with joules per query vs. DuckDB | High: the unique SOTA claim | M | ClickBench and TPC-H subsets, fleet runner |
| 6 | pgwire COPY (bulk ingest from psql `\copy`, pandas `to_sql`, ETL tools) | High | S–M | CopyIn/CopyOut text/CSV |
| 7 | Arrow Flight SQL endpoint | Med-high (ADBC/JDBC/BI) | M | arrow-flight over the existing tonic |
| 8 | JS/TS + WASM package, then Go/Java | Medium | M | npm over HTTP; node-postgres already works through pgwire once #2 holds |
| 9 | Published ann-benchmarks recall@k/QPS for HNSW and HDC, with filtering | Medium (vector credibility) | M | Harness item #5 |
| 10 | Iceberg/Delta/DuckLake attach; LangChain/LlamaIndex adapters; published OpenAI/Anthropic tool schemas from MCP | Medium | M each | After #4 (Parquet) |

Strengths no competitor matches: measured per-device energy receipts, branching plus time travel plus Raft/2PC/sharding in one binary, many query languages, HDC, and MCP built in. All of these stay.
