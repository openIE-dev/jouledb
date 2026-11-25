# Polyglot First-Party SDKs (`13+` Languages & `WASM/WASI`)

`JouleDB v1.0.0` ships first-party, zero-bloat client SDKs directly in the repository (`crates/joule-db-client`, `python/`, and `sdks/*`, Pillar `35`), certified live across **13+ compiled and interpreted programming languages**.

| Language / Runtime | Package / Location | Highlights |
| :--- | :--- | :--- |
| **Rust** | `crates/joule-db-client` & `joule-db-core` | Native embedded `Engine` (`#![deny(unsafe_code)]`) + async `HttpFabricClient` |
| **Python (`>= 3.9`)** | `python/` (`jouledb 1.0.0`) | GIL-free `PyO3` column decoder -> `pyarrow.Table`, `.to_polars()`, `.to_pandas()`, `unsilo()`, `migrate_run()`, `trust_audit()` |
| **TypeScript / Node / Deno / Bun** | `sdks/typescript` (`@openie/jouledb 1.0.0`) | Zero-dependency ES module + strict `.d.ts` types; runs in Node 18+, Deno, Bun, Cloudflare Workers & Browsers |
| **Go (`>= 1.22`)** | `sdks/go` (`github.com/openIE-dev/jouledb/sdks/go`) | Context-aware `jouledb.NewClient`, `Query`, `MigrateCertify`, `TrustAudit`, `PowerhouseSuite` |
| **C# / .NET (`net8.0+`)** | `sdks/dotnet` (`OpenIE.JouleDB 1.0.0`) | Async `HttpClient` + `System.Text.Json` NuGet library (`dotnet pack`) |
| **Swift 6 (`macOS` / `iOS` / `Linux`)** | `sdks/swift` (`JouleDBKit 1.0.0`) | Concurrency-safe `actor JouleDBClient` SwiftPM package |
| **Java (`17+`)** | `sdks/java` (`dev.openie:jouledb-client:1.0.0`) | Zero-external-dependency `java.net.http.HttpClient` Maven artifact |
| **C99 / C++20** | `sdks/c/jouledb_universal.h` & `crates/joule-db-c` | Single-header POSIX/Winsock client + native C ABI FFI (`libjouledb`) |
| **WASM / WASI 0.2 & Browser** | `crates/joule-db-browser` & `joule-db-edge` | `wasm32-wasip1`, `wasm32-wasip2`, Browser `OPFS + WebGPU`, and `#![no_std]` MCU |
| **Ruby / Julia / Lua / R / BI** | Standard HTTP JSON, `pgwire :8082`, Arrow Flight `:8084`, `joule-db-odbc` | Zero-wrapper native wire access |

---

## Quick Polyglot Examples

### Python (`1.0.0`)
```python
import jouledb

db = jouledb.connect("http://127.0.0.1:8080")
res = db.query("SELECT region, SUM(revenue) FROM sales GROUP BY region")
df = res.to_polars()
print(res.receipt["energy_joules"], db.v1_powerhouse_suite()["all_36_pillars_passed"])
```

### TypeScript / Node / Deno / Bun (`@openie/jouledb 1.0.0`)
```ts
import { connect } from "@openie/jouledb";

const db = connect("http://127.0.0.1:8080");
const q = await db.query("SELECT 42 AS answer");
const cert = await db.migrateCertify();
```

### Go (`v1.0.0`)
```go
client := jouledb.NewClient("http://127.0.0.1:8080", "")
res, err := client.Query(ctx, "SELECT * FROM sensors LIMIT 10")
```
