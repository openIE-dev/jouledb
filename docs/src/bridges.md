# Universal Compatibility Bridges (`10` DBs + `7` AI Protocols)

Global compatibility reduces the friction of change management. Existing applications, BI dashboards, vector pipelines, observability collectors, and AI agent frameworks can point their connection strings at `JouleDB v1.0.0` (`crates/joule-db-server/src/universal_bridge.rs`, Pillar `29`) **without rewriting client logic**.

---

## 1. Ten Database Wire & REST Bridges

| Ecosystem | JouleDB Endpoint / Port | Verified Client Compatibility |
| :--- | :--- | :--- |
| **1. PostgreSQL (`pgwire` v3)** | `TCP :8082` (`port + 2`) | `psql`, `psycopg`, `SQLAlchemy`, `asyncpg`, `node-postgres`, JDBC |
| **2. Redis / Valkey (`RESP2/3`)** | Native `RESP` state machine | `PING`, `SET`, `GET`, `INCR`, `HSET`, `EXPIRE`, `DEL` |
| **3. Arrow Flight SQL + Parquet** | `gRPC :8084` (`port + 4`) | Zero-copy `pyarrow.flight`, `polars`, `ADBC`, `JDB_IPC1` |
| **4. DuckLake + Iceberg v3** | `/api/v1/ducklake/*` | Time-travel `AS OF` snapshots & columnar Lakehouse catalogs |
| **5. SQLite3 / C99 FFI / ODBC** | `crates/joule-db-c` & `odbc` | Embedded `libjouledb`, `jouledb_universal.h`, BI ODBC tools |
| **6. Elasticsearch / OpenSearch** | `/_compat/elastic/{idx}/_doc`, `_bulk`, `_search` | Elastic REST ingestion & `BM25` search clients |
| **7. MongoDB Atlas Data API** | `/_compat/mongo/action/insertOne`, `find` | Document CRUD & filter projection without `mongod` |
| **8. Qdrant Vector REST** | `/_compat/qdrant/collections/{c}/points/*` | Drop-in `qdrant-client` upsert & cosine Top-K search |
| **9. Pinecone Data Plane** | `/_compat/pinecone/vectors/upsert`, `/query` | Serverless `pinecone` namespace vector workflows |
| **10. InfluxDB v2 + OTLP + CDC** | `/_compat/influx/api/v2/write`, `/v1/traces`, `/v1/logs`, `/v1/metrics`, `/api/v1/bridge/cdc_ingest` | Telegraf, OpenTelemetry SDKs, and Debezium MySQL/PG/Oracle CDC |

---

## 2. Seven AI & Agentic Protocol Hubs (`54` MCP Tools)

| AI Ecosystem | JouleDB Endpoint | Capability |
| :--- | :--- | :--- |
| **1. OpenAI API** | `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/transcriptions` | 4-tier cascade, SSE `x_jouledb_token_nj`, `XGrammar` JSON/tools, Whisper WAV |
| **2. Anthropic Messages** | `POST /v1/messages` | Claude SDK tool-use + `45 nJ` inline `JoulePulse` firewall |
| **3. Google Gemini `v1beta`** | `POST /v1beta/models/{m}:generateContent` | `google-genai` `contents`/`candidates` wire compatibility |
| **4. Ollama Local CLI** | `/api/tags`, `/api/generate` | Drop-in local `.jsyn` model serving |
| **5. MCP 2026 (Streamable HTTP & Stdio)** | `/mcp`, `/mcp/sse`, `/mcp/messages` | **`54` built-in database, `.jsyn`, Amorphic, Migration, Trust & Frontier tools** |
| **6. Google A2A (`Agent2Agent v1.0`)** | `GET /.well-known/agent.json`, `POST /a2a` | Federated agent discovery card & `tasks/send` orchestration |
| **7. Cohere / Jina Rerank** | `POST /v1/rerank` | Cross-encoder & `TriModal` passage reranking with `nJ` receipts |

```bash
jouledb bridge
curl -s http://127.0.0.1:8080/api/v1/bridge/matrix
```
