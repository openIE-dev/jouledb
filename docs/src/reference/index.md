# API, Query & 54-Tool MCP Reference (`v1.0.0`)

## 1. Seven Native Query Languages (`docs/jouledb/QUERY-*.md`)
- [SQL (`SQL:2023` Window, `CUBE`, `WITH RECURSIVE`, `SEM_*` & Runtime `AQE`)](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-SQL.md)
- [Cypher Property Graph](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-CYPHER.md) · [Datalog](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-DATALOG.md) · [SPARQL](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-SPARQL.md) · [Gremlin](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-GREMLIN.md) · [CQL](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-CQL.md) · [GraphQL](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-GRAPHQL.md)
- [Time-Series, `SigQL` & Causal `ASOF`](https://github.com/openIE-dev/jouledb/blob/main/docs/jouledb/QUERY-TIMESERIES.md) · [Bitemporal Time-Travel & Microsecond `PITR`](./pitr-and-abi.md)

## 2. Primary HTTP & Wire Endpoints (`v1.0.0`)
- **Core SQL, KV & CoW Branching:** `POST /api/v1/query`, `GET|POST|DELETE /api/v1/keys/{key}`, `GET|POST /api/v1/branches`, `POST /api/v1/branches/{name}/merge`
- **Amorphous Continuum & MorphRules:** `POST /api/v1/amorphic/unsilo`, `POST /api/v1/amorphic/morph_rules`, `POST /api/v1/amorphic/stream_scale`
- **Zero-Downtime Migration & `v1.0` GA Certification:** `POST /api/v1/migrate/run`, `POST /api/v1/migrate/certify`, `POST /api/v1/certify/v1_abi_pitr`, `POST /api/v1/bench/sql2023_powerhouse`, `POST /api/v1/certify/geo_raft_jepsen`, `POST /api/v1/bench/v1_powerhouse`
- **Universal 10-DB + 7-AI Bridges:** `/_compat/elastic/*`, `/_compat/mongo/action/*`, `/_compat/qdrant/*`, `/_compat/pinecone/*`, `/_compat/influx/api/v2/write`, `/api/v1/bridge/cdc_ingest`, `/v1/chat/completions`, `/v1/messages`, `/v1beta/models/{m}:generateContent`, `/a2a`, `/v1/rerank`
- **Progressive Trust & SBOM:** `POST /api/v1/trust/compliance_audit`, `GET /api/v1/trust/sbom`, `GET /api/v1/deploy/iac_blueprint`
- **`.jsyn` Living Model Weights & `JouleKube`:** `GET /api/v1/synapse/status`, `POST /api/v1/synapse/hf/pull`, `POST /api/v1/synapse/bench/rank_sweep`, `POST /api/v1/kube/apply`

## 3. MCP 2026 Server (`54` Tools on `/mcp`, `/mcp/sse`, or `stdio`)
- **`24` Core Tools (`tools/list`):** `db.query`, `db.insert`, `db.list_tables`, `db.describe_table`, `db.create_table`, `db.drop_table`, `db.branch_create`, `db.branch_list`, `db.branch_merge`, `db.branch_delete`, `db.vector_search`, `db.vector_upsert`, `db.tenant_create`, `db.tenant_list`, `db.memory_store`, `db.memory_recall`, `db.schema_inspect`, `db.energy_budget`, `db.status`, `db.workflow_create`, `db.workflow_run`, `db.workflow_status`, `db.queue_publish`, `db.queue_consume`.
- **`30` SOTA & `v1.0.0` Powerhouse Extensions (`includeAgenticExtensions: true`):** Includes `amorphic.unsilo_morph`, `amorphic.morph_rules_and_lens`, `amorphic.stream_corpus_scale`, `bridge.ecosystem_certify`, `trust.compliance_audit`, `trust.sbom_cyclonedx`, `migrate.shadow_cutover`, `core.v1_abi_and_pitr`, `query.sql2023_aqe_suite`, `cluster.jepsen_geo_raft`, `deploy.iac_blueprint`, `bench.v1_powerhouse_36`, `sovereign.*`, and `frontier.*`.

## 4. Specifications & Whitepaper
- [Whitepaper `v1.0.0` (November 2026)](https://github.com/openIE-dev/jouledb/blob/main/docs/WHITEPAPER-JOULEDB-2026-11.md)
- [`.jsyn` Living-Weights Binary Spec](https://github.com/openIE-dev/jouledb/blob/main/docs/JSYN-LIVING-WEIGHTS-SPEC.md)
- [HDC 15-Primitive Reference](https://github.com/openIE-dev/jouledb/blob/main/docs/MGAI-HDC-REFERENCE.md)
