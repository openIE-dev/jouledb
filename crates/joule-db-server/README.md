# joule-db-server

Server implementation for JouleDB — networking, replication, multi-protocol wire surface.

`joule-db-server` is the daemon (binary name `jouledb`) that puts the core engine on a wire. It implements five protocol surfaces in parallel, multi-node clustering via Raft, distributed transactions via two-phase commit, sharding, read replicas, SCRAM auth, RBAC, backup/restore, tenancy, observability, and edge-PoP routing.

This is the largest crate in the JouleDB ecosystem — **~182K LOC, 6,202 tests** across `src/`. The full module surface is documented below.

## Protocols (all default-on unless marked)

| Surface | Module | Default | Purpose |
|---|---|---|---|
| **JWP** (Joule Wire Protocol) | [`jwp_server.rs`](src/jwp_server.rs) | yes | Binary, energy-aware protocol — the JouleDB-native wire. TCP / WebSocket. |
| **pgwire** | [`pgwire.rs`](src/pgwire.rs) | yes | PostgreSQL v3 wire — SCRAM auth, prepared statements, type-OID-compatible portals. Lets Excel/Tableau/Power BI/etc. connect via stock PostgreSQL drivers. |
| **WebSocket** | [`websocket.rs`](src/websocket.rs) | yes | Real-time subscriptions, bidirectional messaging |
| **HTTP** | [`binary_protocol.rs`](src/binary_protocol.rs), [`dynamic_routes.rs`](src/dynamic_routes.rs), [`vector_routes.rs`](src/vector_routes.rs) | yes | REST + HTTP/2 + JSON ops surface |
| **WebTransport** | [`webtransport.rs`](src/webtransport.rs) | opt-in | HTTP/3 + QUIC |
| **MCP** | [`mcp_bridge.rs`](src/mcp_bridge.rs), [`mcp_transport.rs`](src/mcp_transport.rs) | yes | Model Context Protocol — exposes DB tools to LLM clients. Separate from the `ask-mcp` surface. |
| **TCP raw** | [`tcp_server.rs`](src/tcp_server.rs) | yes | Low-level multiplexed TCP listener |
| **Redis RESP** | [`resp.rs`](src/resp.rs) | yes | RESP3 compatibility for KV workloads |

## Clustering & distribution

| Module | Role |
|---|---|
| [`raft.rs`](src/raft.rs), [`raft_server.rs`](src/raft_server.rs), [`raft_transport.rs`](src/raft_transport.rs) | Raft consensus — leader election, log replication, snapshots |
| [`two_phase_commit.rs`](src/two_phase_commit.rs) | 2PC coordinator for cross-shard transactions |
| [`sharding.rs`](src/sharding.rs) | Hash and range sharding strategies |
| [`read_replica.rs`](src/read_replica.rs) | Eventually-consistent read followers |
| [`replication.rs`](src/replication.rs) | Replication topology + lag tracking |
| [`distributed_query.rs`](src/distributed_query.rs) | Cross-shard query planning |
| [`edge_pop.rs`](src/edge_pop.rs) | Edge points-of-presence routing |
| [`scale_to_zero.rs`](src/scale_to_zero.rs) | Idle-shutdown / cold-start orchestration |
| [`tenant.rs`](src/tenant.rs) | Multi-tenant isolation |

## Query engine integration (per-language executors)

| Module | Language |
|---|---|
| [`cql_executor.rs`](src/cql_executor.rs) | Cassandra CQL |
| [`cypher_executor.rs`](src/cypher_executor.rs) | Neo4j Cypher |
| [`datalog_executor.rs`](src/datalog_executor.rs) | Datalog |
| [`graphql_executor.rs`](src/graphql_executor.rs) | GraphQL |
| [`gremlin_executor.rs`](src/gremlin_executor.rs) | Apache TinkerPop Gremlin |
| [`sparql_executor.rs`](src/sparql_executor.rs) | RDF SPARQL |
| [`sigql_executor.rs`](src/sigql_executor.rs) | sigql |
| [`energy_executor.rs`](src/energy_executor.rs) | Energy-receipt-instrumented executor wrapper |
| [`query.rs`](src/query.rs) | SQL dispatch entry point (delegates to [`joule-db-query`](../joule-db-query/)) |

## Auth & security

| Module | Role |
|---|---|
| [`auth.rs`](src/auth.rs) | API keys, JWT, session tokens |
| [`scram.rs`](src/scram.rs) | SCRAM-SHA-256 password auth (pgwire-compatible) |
| [`rbac.rs`](src/rbac.rs) | Role-based access control |
| [`security.rs`](src/security.rs) | TLS, cert pinning, secret handling |
| [`hrp_security.rs`](src/hrp_security.rs), [`hrp_erasure.rs`](src/hrp_erasure.rs) | High-resilience-pool: secret erasure, key derivation |
| [`audit.rs`](src/audit.rs) | Audit log emission |

## Indexes & operators

| Module | Role |
|---|---|
| [`vector_index.rs`](src/vector_index.rs) | k-NN / cosine / dot-product indexes |
| [`spatial_index.rs`](src/spatial_index.rs) | R-tree / quadtree / grid |
| [`gin_index.rs`](src/gin_index.rs) | Generalised inverted index |
| [`fts_analyzer.rs`](src/fts_analyzer.rs) | Full-text tokenisation pipeline |
| [`json_ops.rs`](src/json_ops.rs) | JSONB-style operators |
| [`mutation_delta.rs`](src/mutation_delta.rs) | Mutation delta tracking for subscriptions |
| [`subscriptions.rs`](src/subscriptions.rs), [`subscription_hdc.rs`](src/subscription_hdc.rs) | Real-time subscription engine; HDC-encoded variant |
| [`workflow.rs`](src/workflow.rs) | Multi-step workflow execution |
| [`langgraph_handlers.rs`](src/langgraph_handlers.rs) | LangGraph checkpoint / message handlers |

## Storage adapters & memory

| Module | Role |
|---|---|
| [`mvcc_adapter.rs`](src/mvcc_adapter.rs) | MVCC bridge to [`joule-db-core`](../joule-db-core/) |
| [`amorphic_adapter/`](src/amorphic_adapter/) | Bridge to [`joule-db-amorphic`](../joule-db-amorphic/) |
| [`holographic_adapter.rs`](src/holographic_adapter.rs) | Bridge to [`joule-db-hdc`](../joule-db-hdc/) holographic store |
| [`features_bridge.rs`](src/features_bridge.rs) | Bridge to [`joule-db-features`](../joule-db-features/) |
| [`ledger_bridge.rs`](src/ledger_bridge.rs) | Bridge to [`joule-db-ledger`](../joule-db-ledger/) |
| [`cxl_memory.rs`](src/cxl_memory.rs) | CXL pooled-memory backend |
| [`pool.rs`](src/pool.rs) | Connection pool |
| [`agent_memory.rs`](src/agent_memory.rs) | Agent-scoped memory store |

## Operations

| Module | Role |
|---|---|
| [`backup.rs`](src/backup.rs) | Full + incremental backups, compression, encryption |
| [`deployment.rs`](src/deployment.rs) | Deployment metadata + topology |
| [`enterprise.rs`](src/enterprise.rs) | Failover manager, load balancer, node health |
| [`operations.rs`](src/operations.rs) | Migrations, consistency checks, reindex, vacuum |
| [`config.rs`](src/config.rs) | Configuration |
| [`health.rs`](src/health.rs) | Health endpoints |
| [`energy.rs`](src/energy.rs) | Energy budget enforcement per query |
| [`logging.rs`](src/logging.rs) | Structured logging |
| [`metrics.rs`](src/metrics.rs) | Prometheus metrics |
| [`observability.rs`](src/observability.rs) | Tracing + telemetry |
| [`request_tracing.rs`](src/request_tracing.rs) | Per-request span tree |
| [`multiplex.rs`](src/multiplex.rs) | Stream multiplexing |
| [`lock_util.rs`](src/lock_util.rs) | Lock primitives + poisoning recovery |

## Verification

| Module | Role |
|---|---|
| [`kani_proofs.rs`](src/kani_proofs.rs) | Kani formal proofs for invariants |
| [`proptest_verify.rs`](src/proptest_verify.rs) | Property-based regression suite |

## Real-time engine

| Module | Role |
|---|---|
| [`realtime/`](src/realtime/) | Real-time subscription delivery engine |

## Feature flags

| Feature | Default | Purpose |
|---|---|---|
| `http` | yes | HTTP/REST API |
| `websocket` | yes | WebSocket subscriptions |
| `jwp` | yes | Joule Wire Protocol |
| `viz` | yes | Visualization hint inference via [`joule-db-viz`](../joule-db-viz/) |
| `replication` | off | Multi-node replication |
| `tls` | off | rustls + TLS |
| `webtransport` | off | HTTP/3 + QUIC |
| `wasm-triggers` | off | wasmtime-backed triggers |
| `adaptive-pool` | off | Adaptive connection pool sizing |

## Binary

Build with `cargo build --release -p joule-db-server`; the binary is named `jouledb`. Default JWP listener port is `9090` (local) or `9000` (cloud, via [`joule-cloud-provisioner`](../joule-cloud-provisioner/)).

## Tests

6,202 `#[test]` / `#[tokio::test]` annotations — the largest test suite in the workspace. Fuzz targets live in [`fuzz/`](./fuzz/). Tests in [`tests/`](./tests/), benches in [`benches/`](./benches/), in-tree backup snapshots in [`backups/`](./backups/).

## See also

- [WHITEPAPER-JOULEDB-2026-04.md](../../docs/WHITEPAPER-JOULEDB-2026-04.md)
- [MGAI-ACP-REFERENCE.md](../../docs/MGAI-ACP-REFERENCE.md) — daemon invocation contract
- `docs/MGAI-JWP-PROTOCOL.md` — wire protocol spec *(in progress)*
- [openapi.yaml](../joule-cloud-api-gateway/openapi.yaml) — cloud API surface
