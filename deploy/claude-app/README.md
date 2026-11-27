# JouleDB Claude App & MCP Bundle (`.mcpb`, `v0.15.0`)

This directory packages **JouleDB** as both:
1. A one-click **Claude Desktop `.mcpb` Bundle** (`manifest_version: "0.3"`), and
2. An interactive **MCP App (`SEP-1865`, `ui://jouledb/studio.html`, `text/html;profile=mcp-app`)** rendering inline inside **Claude Desktop**, **claude.ai**, **Claude Cowork**, **VS Code**, and **Google Antigravity**.

## Quick Install Options

### Option A — One-Click `.mcpb` Bundle in Claude Desktop
Download `jouledb-0.15.0-aarch64-apple-darwin.mcpb` (or `x86_64`) from [GitHub Releases](https://github.com/openIE-dev/jouledb/releases/tag/v0.15.0) and double-click or drag it into **Claude Desktop -> Settings -> Extensions**.

### Option B — Claude Code CLI (`claude mcp add`)
```bash
cargo binstall jouledb
claude mcp add jouledb -- jouledb mcp --data ~/.jouledb/claude
```

### Option C — `claude_desktop_config.json` (Any OS)
```json
{
  "mcpServers": {
    "jouledb": {
      "command": "jouledb",
      "args": ["mcp", "--data", "/path/to/jouledb-data"]
    }
  }
}
```

### Option D — Remote Streamable HTTP Connector (`claude.ai` / Cowork)
Start `jouledb --port 8080 --data ./joule-data` and point your MCP Streamable HTTP connector at `/mcp`.
