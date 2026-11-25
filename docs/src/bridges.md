# Wire & HTTP Compatibility Adapters (`bridges.md`)

JouleDB implements wire-protocol and HTTP JSON translation routes (`crates/joule-db-server/src/universal_bridge.rs` and `pgwire`) so common database drivers and AI HTTP clients can talk to a local or self-hosted JouleDB instance for supported subsets.

## 1. Live-Tested Wire Protocol (`tests/clients/`)

- **PostgreSQL v3 Wire (`pgwire` on `port + 2`, default `:8082`):** Exercised in CI (`pg-clients.yml` and `tests/clients/`) against real `psql`, Python `psycopg3`, `SQLAlchemy`, Node.js `pg` (`node-postgres`), and Java `JDBC` drivers (`CREATE TABLE`, `COPY`, `INSERT`, `SELECT`, aggregates). Note: JouleDB implements a practical SQL subset, not full PostgreSQL system catalogs (`pg_catalog` completeness) or PL/pgSQL.

## 2. HTTP Compatibility Adapters (`universal_bridge.rs`)

These Axum routes translate common JSON/line-protocol payloads into JouleDB tables, vector collections, or Amorphic records (verified via HTTP integration tests in `tests/wave30_universal_ecosystem_bridge_and_trust_compliance_e2e.rs`):

- **Document & Search HTTP Subsets:** `/_compat/elastic/{index}/_doc`, `_bulk`, `_search`; `/_compat/mongo/action/{insertOne,insertMany,find,aggregate}`.
- **Vector HTTP Subsets:** `/_compat/qdrant/collections/{name}/points` (upsert & cosine search); `/_compat/pinecone/vectors/upsert` & `/query`.
- **Telemetry & CDC Ingest:** `/_compat/influx/api/v2/write` (line protocol), `/v1/logs` & `/v1/metrics` (OTLP JSON), `/api/v1/bridge/cdc_ingest` (Debezium JSON envelopes).
- **AI / Agent HTTP Subsets:** OpenAI (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`), Anthropic (`/v1/messages`), Gemini (`/v1beta/models/{m}:generateContent`), Ollama (`/api/generate`, `/api/tags`), MCP JSON-RPC (`/mcp` — `49` tools), A2A (`/.well-known/agent.json`, `/a2a`), and `/v1/rerank`.

> **Scope note:** `jouledb bridge` (`GET /api/v1/bridge/matrix`) runs a fast in-process 5-fragment entity-merge + 4D vector lookup unit smoke check (`in_process_entity_merge_smoke_passed`), not external third-party server benchmarks.
