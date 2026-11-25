# Client Libraries & Minimal HTTP Wrappers (`v0.6.0`)

JouleDB can be accessed through standard PostgreSQL drivers (`pgwire` on `:8082`), the Rust client/embedded crates, the Python `PyO3` package, or six lightweight zero-dependency HTTP wrapper files under `sdks/`.

## Honest Maturity & Registry Status Matrix

| Language | Path in Repo | Size / Scope | Public Registry Status (`v0.6.0`) |
| :--- | :--- | :--- | :--- |
| **PostgreSQL Drivers** (`Python`, `Node`, `Java`, `Go`, `C#`, `Ruby`) | Standard `psycopg3`, `node-pg`, `JDBC`, `pgx`, `Npgsql` against `:8082` | Production client ecosystems speaking PostgreSQL v3 wire (`tests/clients/`) | Upstream registries |
| **Rust** | `crates/joule-db-core`, `joule-db-local`, `joule-db-client` | Full embedded storage engine + async `HttpFabricClient` | Workspace source (`jouledb` `0.6.0` on `crates.io` is the CLI binary-fetcher stub) |
| **Python (`>= 3.9`)** | `python/` (`PyO3` + HTTP client) | Native `cp39-abi3` Rust extension + Arrow/Polars helpers | Wheels attached to **GitHub Releases** (**not** yet published on PyPI) |
| **TypeScript / JS** | `sdks/typescript/` (`218` LOC) | Minimal `fetch` wrapper over HTTP `/api/v1/*` | In-repo source (**not** published on npm) |
| **Go** | `sdks/go/jouledb.go` (`163` LOC) | Minimal `net/http` wrapper over HTTP `/api/v1/*` | In-repo Go module |
| **C99 / C++** | `crates/joule-db-c` & `sdks/c/jouledb_universal.h` (`110` LOC) | Native `cdylib` FFI + minimal POSIX/Winsock HTTP header | In-repo source |
| **Swift** | `sdks/swift/` (`71` LOC) | Minimal `URLSession` `actor` wrapper | In-repo SwiftPM source |
| **Java (`17+`)** | `sdks/java/` (`63` LOC) | Minimal `java.net.http.HttpClient` wrapper | In-repo Maven source |
| **C# / .NET** | `sdks/dotnet/` (`56` LOC) | Minimal `HttpClient` wrapper | In-repo `.csproj` (**not** published on NuGet) |
