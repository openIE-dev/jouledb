# API, Query & 58-Tool MCP Reference (`v0.13.0`)

## 1. Query Languages (`docs/jouledb/QUERY-*.md`)
- [SQL Subset (`SELECT`, Joins, Window Frames, `CUBE`, `WITH RECURSIVE`, `COPY`)](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-SQL.md)
- [Cypher](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-CYPHER.md) · [Datalog](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-DATALOG.md) · [SPARQL](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-SPARQL.md) · [Gremlin](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-GREMLIN.md) · [CQL](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-CQL.md) · [GraphQL](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-GRAPHQL.md)
- [Time-Series, `SigQL` & Causal `ASOF`](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-TIMESERIES.md) · [TLV Framing & WAL `PITR`](../pitr-and-abi.md)

## 2. Primary HTTP & Wire Endpoints (`v0.13.0`)
- **Core SQL, KV & CoW Branching:** `POST /api/v1/query`, `GET|POST|DELETE /api/v1/keys/{key}`, `GET|POST /api/v1/branches`, `POST /api/v1/branches/{name}/merge`
- **Amorphous Continuum & MorphRules:** `POST /api/v1/amorphic/unsilo_morph`, `POST /api/v1/amorphic/morph_rules_and_lens`, `POST /api/v1/amorphic/stream_corpus_scale`
- **Import & In-Process Subsystem Smoke Checks:** `POST /api/v1/migrate/run`, `POST /api/v1/certify/v1_abi_pitr`, `POST /api/v1/bench/sql2023_powerhouse`, `POST /api/v1/certify/geo_raft_jepsen` *(5-struct in-memory sim)*, `POST /api/v1/bench/v1_powerhouse` *(6-check smoke runner)*
- **HTTP Compatibility Adapters:** `/_compat/elastic/*`, `/_compat/mongo/action/*`, `/_compat/qdrant/*`, `/_compat/pinecone/*`, `/_compat/influx/api/v2/write`, `/api/v1/bridge/cdc_ingest`, `/v1/chat/completions`, `/v1/messages`, `/v1beta/models/{m}:generateContent`, `/a2a`, `/v1/rerank`
- **Unaudited Security Control Self-Tests & SBOM:** `POST /api/v1/trust/compliance_audit`, `GET /api/v1/trust/sbom`
- **`.jsyn` Weights & Rank-Sweep:** `GET /api/v1/synapse/status`, `POST /api/v1/synapse/hf/pull`, `POST /api/v1/synapse/bench/rank_sweep`

## 3. MCP JSON-RPC Server & `SEP-1865` Studio UI (`50` Tools on `/mcp` or `jouledb mcp` `stdio`)
- **Interactive MCP App Resource (`resources/list`, `resources/read`):** `ui://jouledb/studio.html` (`text/html;profile=mcp-app`, strict `default-src 'none'` CSP; see [Claude Apps & MCP Studio](../claude-apps.md)).
- **`25` Core Database Tools (`tools/list`):** `db.studio` *(interactive Studio UI entrypoint)*, `db.query`, `db.insert`, `db.list_tables`, `db.describe_table`, `db.create_table`, `db.drop_table`, `db.branch_create`, `db.branch_list`, `db.branch_merge`, `db.branch_delete`, `db.vector_search`, `db.vector_upsert`, `db.tenant_create`, `db.tenant_list`, `db.memory_store`, `db.memory_recall`, `db.schema_inspect`, `db.energy_budget`, `db.status`, `db.workflow_create`, `db.workflow_run`, `db.workflow_status`, `db.queue_publish`, `db.queue_consume`.
- **`25` Extended Tools (`includeAgenticExtensions: true`):** `db.branch_diff`, `db.agentfs_{write,read,list}`, `db.memory_inhibit`, `db.explain_energy`, `sql.dialect_repair`, `robotics.stl_trajectory_clamp`, `db.synthetic_columnar_smoke`, `amorphic.*`, `sovereign.*`, `bridge.entity_merge_smoke`, `trust.security_controls_self_test`, `trust.sbom_cyclonedx`, `migrate.shadow_cutover`, `core.tlv_and_pitr_smoke`, `query.window_cube_aqe_smoke`, `cluster.in_memory_partition_sim`, `deploy.iac_blueprint`, `bench.subsystem_smoke`.
