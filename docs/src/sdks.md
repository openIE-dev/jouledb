# Client Libraries, Claude `.mcpb` Bundles & Minimal HTTP Wrappers (`v0.13.0`)

JouleDB can be accessed through standard PostgreSQL drivers (`pgwire` on `:8082`), **Claude Apps / MCP Apps (`jouledb mcp` & `/mcp`, `ui://jouledb/studio.html`)**, the Rust client/embedded crates, the Python `PyO3` `cp39-abi3` package, or six lightweight zero-dependency HTTP wrapper libraries under `sdks/`.

## Honest Maturity & Registry Status Matrix (`v0.13.0`)

| Language / Host | Path in Repo | Size / Scope | Distribution & Public Registry Status (`v0.13.0`) |
| :--- | :--- | :--- | :--- |
| **PostgreSQL Drivers** (`Python`, `Node`, `Java`, `Go`, `C#`, `Ruby`) | Standard `psycopg3` (Simple & Extended Pipeline), `node-pg`, `JDBC`, `pgx`, `Npgsql` against `:8082` | Production client ecosystems speaking PostgreSQL v3 wire (`tests/clients/`) | Upstream registries |
| **Claude Desktop / Claude Code / Antigravity (`MCP Apps`)** | `deploy/claude-app/`, `deploy/antigravity-plugin/`, `jouledb mcp` | Built-in `SEP-1865` Studio UI (`ui://jouledb/studio.html`) + `50` MCP tools | **`.mcpb` bundles attached to GitHub Release `v0.13.0`** (`manifest_version: "0.3"`) |
| **Rust** | `crates/joule-db-core`, `joule-db-local`, `joule-db-client` | Full embedded storage engine + async `HttpFabricClient` | Workspace source (`jouledb` `0.13.0` on `crates.io` is the CLI binary-fetcher stub) |
| **Python (`>= 3.9`, `cp39-abi3`)** | `python/` (`PyO3` + HTTP client) | Native `cp39-abi3` Rust extension + Arrow/Pandas/Polars helpers | **Prebuilt `.whl` wheels (macOS ARM64/x86_64 & `manylinux_2_34_x86_64`) on GitHub Release `v0.13.0`** (**not** yet published on PyPI) |
| **TypeScript / JS** | `sdks/typescript/` (`218` LOC) | Minimal zero-dep `fetch` wrapper over HTTP `/api/v1/*` | **`openie-jouledb-0.13.0.tgz` attached to GitHub Release `v0.13.0`** (**not** yet published on npm) |
| **Go** | `sdks/go/jouledb.go` (`163` LOC) | Minimal `net/http` wrapper over HTTP `/api/v1/*` | In-repo Go module (`v0.13.0`) |
| **C99 / C++** | `crates/joule-db-c` & `sdks/c/jouledb_universal.h` (`110` LOC) | Native `cdylib` FFI + minimal POSIX/Winsock HTTP header | In-repo source (`v0.13.0`) |
| **Swift** | `sdks/swift/` (`71` LOC) | Minimal `URLSession` `actor` wrapper | In-repo SwiftPM source (`v0.13.0`) |
| **Java (`17+`)** | `sdks/java/` (`63` LOC) | Minimal `java.net.http.HttpClient` wrapper | In-repo Maven source (`v0.13.0`) |
| **C# / .NET** | `sdks/dotnet/` (`56` LOC) | Minimal `HttpClient` wrapper | In-repo `.csproj` (`v0.13.0`; **not** published on NuGet) |

### Installing Prebuilt `v0.13.0` Python Wheel or TypeScript Tarball from GitHub Releases

```bash
# Python (>= 3.9, macOS Apple Silicon/Intel or Linux x86_64 glibc >= 2.34)
pip install https://github.com/openIE-dev/jouledb/releases/download/v0.13.0/jouledb-0.13.0-cp39-abi3-manylinux_2_34_x86_64.whl

# Node.js / Bun / Deno
npm install https://github.com/openIE-dev/jouledb/releases/download/v0.13.0/openie-jouledb-0.13.0.tgz
```
