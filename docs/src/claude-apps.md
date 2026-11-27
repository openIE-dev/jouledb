# Claude Apps, MCP Apps (`SEP-1865`) & Antigravity Studio UI (`v0.13.0`)

JouleDB `v0.13.0` ships a built-in, zero-CDN interactive **MCP App (`SEP-1865` / `@modelcontextprotocol/ext-apps`)** directly inside the single `jouledb` binary (`ui://jouledb/studio.html`, MIME `text/html;profile=mcp-app`).

When connected to **Claude** (Claude Desktop, Claude on the web / mobile via remote MCP connectors, Claude Cowork, or Claude Code), **Google Antigravity** (inline chat cards, `aux_pane` side panel, and `global` sidebar view), **ChatGPT**, **VS Code**, **Cursor**, or **Goose**, calling `db.studio`, `db.query`, `db.branch_list`, or `db.schema_inspect` renders a live sandboxed **JouleDB Studio** workbench directly inside the conversation — while simultaneously returning compact token-efficient text + `structuredContent` for hosts without UI rendering.

---

## 1. Architecture & `SEP-1865` Compliance

The built-in Studio UI (`crates/joule-db-server/src/mcp_app_ui.rs`) is served over **both** MCP transports from the same `jouledb` binary:

1. **Local `stdio` MCP (`jouledb mcp --data <dir>`):** Opens a durable local `DurableAmorphicStore` + `SimpleQueryExecutor` + `DatabaseToolHandler` over JSON-RPC 2.0 `stdin`/`stdout` (`notifications/initialized` silent handling compliant with Claude Desktop & Claude Code).
2. **Remote Streamable HTTP MCP (`POST /mcp` on `jouledb --port 8080`):** Serves remote Claude Connector / multi-agent HTTP clients with session headers (`Mcp-Session-Id`) and SSE streaming.

### Security & Sandbox Posture
- **Strict Content Security Policy:** `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:;">` — zero external CDN requests, zero third-party scripts, zero telemetry beacons.
- **Standard `SEP-1865` Lifecycle Handshake:** Sends `ui/initialize` (`protocolVersion: "2026-01-26"`) over `window.parent.postMessage`, emits `ui/notifications/initialized`, tracks `ui/notifications/tool-input`, `ui/notifications/tool-result`, and `ui/notifications/host-context-changed` (`data-theme` + `--color-*` host CSS custom properties), and reports iframe height via `ResizeObserver` (`ui/notifications/size-changed`).
- **Interactive Capabilities Inside the Sandboxed Iframe:**
  - **SQL & Vector/Graph Workbench:** Run arbitrary SQL queries (`Cmd/Ctrl+Enter`) via `tools/call` -> `db.query`, sort any result column client-side, and switch between tabular grid and automatic inline distribution bar charts.
  - **Live Schema & Sample Inspector:** Browse tables via `db.schema_inspect` and one-click sample (`SELECT * FROM <tbl> LIMIT 25`).
  - **Copy-on-Write (`O(1)`) Branch Manager:** List, fork (`db.branch_create`), switch, and merge (`db.branch_merge`) isolated database branches for safe agentic sandboxing before mutating `main`.
  - **Thermodynamic Receipt Telemetry & Model Context Sync:** Displays latency (`ms`), row counts, active branch, and joule telemetry (`reported_j` hardware RAPL / `powermetrics` when metered, or `est_j` analytical fallback), plus one-click **"Pin to Claude Context"** (`ui/update-model-context`) and **"Ask Claude to Analyze"** (`ui/message`).

---

## 2. Four Ways to Connect JouleDB to Claude & Antigravity

### Option A: One-Click Claude Desktop Bundle (`.mcpb`, `manifest_version: "0.3"`)

Each GitHub Release (`v0.13.0+`) attaches platform-native `.mcpb` bundles (`jouledb-0.13.0-aarch64-apple-darwin.mcpb`, `jouledb-0.13.0-x86_64-apple-darwin.mcpb`, `jouledb-0.13.0-x86_64-unknown-linux-gnu.mcpb`):

1. Download the `.mcpb` file for your OS/architecture from [GitHub Releases (`v0.13.0`)](https://github.com/openIE-dev/jouledb/releases/tag/v0.13.0).
2. Double-click the `.mcpb` file (or drag it into **Claude Desktop -> Settings -> Extensions**).
3. Choose an optional local data directory (defaults to `~/.jouledb/claude-studio`).
4. Ask Claude: *"Open JouleDB Studio, create a `sensors` table with 5 rows, fork branch `exp-threshold`, and compare aggregates."*

*(Source manifest: [`deploy/claude-app/manifest.json`](https://github.com/openIE-dev/jouledb/blob/main/deploy/claude-app/manifest.json))*

### Option B: Claude Code CLI (`claude mcp add`)

```bash
# Install jouledb v0.13.0 via cargo-binstall (or GitHub Release tarball)
cargo binstall -y jouledb

# Register local durable JouleDB MCP server + Studio App in Claude Code
claude mcp add jouledb -- jouledb mcp --data ~/.jouledb/claude-code
```

### Option C: Claude Desktop Manual `claude_desktop_config.json`

```json
{
  "mcpServers": {
    "jouledb": {
      "command": "jouledb",
      "args": ["mcp", "--data", "/Users/you/.jouledb/studio"]
    }
  }
}
```

### Option D: Remote Streamable HTTP Connector (Claude Web / Cowork / Mobile) & Google Antigravity Plugin

For remote Claude Connectors (`Settings -> Connectors -> Add Custom Connector`) or multi-user deployments, expose `https://<your-host>/mcp` in front of `jouledb --port 8080 --data ./data`.

For **Google Antigravity**, copy or symlink [`deploy/antigravity-plugin/`](https://github.com/openIE-dev/jouledb/tree/main/deploy/antigravity-plugin) — `db.studio` declares `com.google.antigravity/ui.entrypoints: ["aux_pane", "global"]` so JouleDB Studio opens both inline in chat and as a docked side-pane or full-screen sidebar panel.

---

## 3. UI-Linked MCP Tools (`50` Total Tools)

| Tool | UI Resource URI | Visibility | Purpose |
| :--- | :--- | :--- | :--- |
| `db.studio` | `ui://jouledb/studio.html` | `["model", "app"]` (`aux_pane`, `global`) | Opens the interactive JouleDB Studio workbench (optionally pre-running `sql` on `branch`) |
| `db.query` | `ui://jouledb/studio.html` | `["model", "app"]` | Executes SQL and renders sortable table + distribution chart + joule receipt |
| `db.branch_list` | `ui://jouledb/studio.html` | `["model", "app"]` | Lists `O(1)` CoW branches (`main` + forks) and hydrates the Studio branch bar |
| `db.schema_inspect` | `ui://jouledb/studio.html` | `["model", "app"]` | Inspects table columns/types and hydrates the Studio schema sidebar |

All remaining `21` core database/branch/capsule/migration tools and `25` agentic extension tools (`db.memory_*`, `db.semantic_cache_*`, `db.fabric_*`, `db.continuum_*`) remain available in the `50`-tool catalog (`includeAgenticExtensions: true`).
