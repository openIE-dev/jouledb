# JouleDB MCP Reference

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-db-server/src/mcp_bridge.rs`](../../crates/joule-db-server/src/mcp_bridge.rs), [`crates/joule-db-server/src/mcp_transport.rs`](../../crates/joule-db-server/src/mcp_transport.rs)
**Distinct from:** [`MGAI-MCP-REFERENCE.md`](../MGAI-MCP-REFERENCE.md) — that doc covers the `ask-mcp` (askdavidc.ai) MCP server. This doc covers the **JouleDB-side MCP bridge** built into `joule-db-server`.

`joule-db-server` exposes Model Context Protocol tools so AI agents (Claude Code, Cursor, Xcode 26.3, OpenAI Agents, etc.) can talk to a JouleDB instance directly. Transport: stdio JSON-RPC or HTTP SSE.

---

## 1. Tools exposed

| Tool | Purpose |
|---|---|
| `db.query` | Execute SQL — `SELECT`, `INSERT`, `UPDATE`, `DELETE`, DDL. Returns rows + energy receipt. |
| `db.get` | KV retrieve by key (compiled to `SELECT`) |
| `db.put` | KV store with optional TTL (compiled to `INSERT`/`UPDATE`) |
| `db.delete` | KV delete by key |
| `db.semantic_search` | Vector similarity search over an indexed column |
| `db.energy` | Energy metrics snapshot — current process, ledger totals |

All tools live in the `db.*` namespace per the shared `inv-mcp-core` trait (`McpToolHandler`).

---

## 2. Transport

### 2.1 stdio JSON-RPC (default for agent integrations)

```bash
jouledb mcp --stdio
```

The server reads JSON-RPC requests from stdin, writes responses to stdout, logs to stderr. Standard MCP framing.

### 2.2 HTTP SSE (for web-based agent platforms)

```bash
jouledb mcp --http :8080/mcp
```

POST JSON-RPC to `/mcp/request`, GET SSE stream from `/mcp/stream`.

---

## 3. Tool invocation examples

### 3.1 `db.query`

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "tools/call",
  "params": {
    "name": "db.query",
    "arguments": {
      "sql": "SELECT name, age FROM users WHERE active = true LIMIT 10",
      "args": [],
      "params": {},
      "limit": 100,
      "explain": false
    }
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "result": {
    "columns": ["name", "age"],
    "rows": [["Alice", 32], ["Bob", 28]],
    "row_count": 2,
    "energy_uwh": 1245,
    "elapsed_ms": 3
  }
}
```

### 3.2 `db.semantic_search`

```json
{
  "method": "tools/call",
  "params": {
    "name": "db.semantic_search",
    "arguments": {
      "table": "articles",
      "embedding_column": "embedding",
      "query_vector": [0.12, 0.45, ...],
      "limit": 5,
      "distance": "cosine"
    }
  }
}
```

### 3.3 `db.energy`

```json
{
  "method": "tools/call",
  "params": { "name": "db.energy", "arguments": {} }
}
```

Response:

```json
{
  "result": {
    "process_joules": 12345.67,
    "ledger_total_joules": 999000.42,
    "queries_per_sec": 84.2,
    "tier_breakdown": { "Lookup": 80.1, "Extract": 15.4, "Reason": 4.5 }
  }
}
```

---

## 4. Configuration in an MCP client

### Claude Code (`mcp.json`)

```json
{
  "mcpServers": {
    "jouledb-prod": {
      "command": "jouledb",
      "args": ["mcp", "--stdio", "--url", "jouledb://my-cluster.jouledb.cloud:9000"],
      "env": { "JOULE_DB_TOKEN": "..." }
    }
  }
}
```

### Cursor / Xcode 26.3 / OpenAI Agents

Same shape — point at the `jouledb mcp --stdio` binary. Each client has its own UI for adding MCP servers.

---

## 5. Energy receipts in MCP responses

Every tool result includes `energy_uwh` in its top-level fields. Wrapping clients (Claude Code, Cursor) typically expose this in their UI as the per-tool-call cost, alongside elapsed time.

---

## 6. Distinction from `ask-mcp`

| Surface | Purpose | Tools |
|---|---|---|
| **`ask-mcp`** ([MGAI-MCP-REFERENCE.md](../MGAI-MCP-REFERENCE.md)) | askdavidc.ai user-facing | `askdavidc_search`, `joule_web_compute`, `joule_web_list` |
| **`db.*` MCP** (this doc) | JouleDB direct access | `db.query`, `db.get`, `db.put`, `db.delete`, `db.semantic_search`, `db.energy` |

`ask-mcp` is the consumer-facing surface (search the joule-search graph, run joule-web compute modules). This `db.*` MCP is the database-direct surface (raw SQL access, KV ops, vector search). Either an agent uses one or the other; sometimes both.

---

## 7. Auth

`jouledb mcp --stdio` typically inherits credentials from the calling process's environment:

| Env var | Purpose |
|---|---|
| `JOULE_DB_TOKEN` | Bearer token for the JouleDB server |
| `JOULE_DB_URL` | Server URL (`jouledb://host:port`) |
| `MCP_ALLOWED_TOOLS` | Comma-separated tool whitelist (e.g. `db.query,db.semantic_search`) |
| `MCP_READ_ONLY` | If `true`, refuse `db.put`, `db.delete`, and write-mode `db.query` |

`MCP_READ_ONLY=true` + `MCP_ALLOWED_TOOLS=db.query,db.semantic_search` is the recommended config for agent setups where you want the AI to read but not mutate.

---

## 8. See also

- [`MGAI-MCP-REFERENCE.md`](../MGAI-MCP-REFERENCE.md) — the askdavidc.ai MCP surface (different tools)
- [`MGAI-JWP-PROTOCOL.md`](../MGAI-JWP-PROTOCOL.md) — JouleDB's native wire protocol (alternative to MCP)
- [`crates/joule-db-server/src/mcp_bridge.rs`](../../crates/joule-db-server/src/mcp_bridge.rs)
- [`crates/joule-db-server/src/mcp_transport.rs`](../../crates/joule-db-server/src/mcp_transport.rs)
- [Anthropic MCP spec](https://spec.modelcontextprotocol.io/)

---

*Drafted 2026-05-18 as wave 3 of the JouleDB documentation parity pass.*
