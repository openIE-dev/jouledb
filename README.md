<p align="center">
  <img src="https://jouledb.org/favicon.svg" width="80" alt="JouleDB" />
</p>

<h1 align="center">JouleDB</h1>

<p align="center">
  <strong>The world's first energy-aware database</strong> — powered by Hyperdimensional Computing (HDC), hardware-adaptive, with real-time J/query telemetry.
</p>

<p align="center">
  <a href="https://jouledb.org">Website</a> · <a href="https://projects.openie.dev">Projects</a> · <a href="https://blog.openie.dev">Blog</a> · <a href="https://openie.dev">Open Interface Engineering</a>
</p>

---

```
                ┌─────────────────────────────────────┐
                │         JSON Document Ingest        │
                └──────────────────┬──────────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
       ▼                           ▼                           ▼
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│  HDC Hologram │           │  B-Tree Index │           │   Columnar   │
│  (Similarity) │           │ (Point/Range) │           │ (Analytics)  │
└──────────────┘           └──────────────┘           └──────────────┘
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │  Energy Monitor │
                          │   (J/query)     │
                          └─────────────────┘
```

JouleDB is a hybrid OLTP/OLAP/semantic database written in **Rust** (Edition 2024). It encodes every record into five concurrent data structures at write time, adapts execution based on hardware power/thermal state, and tracks energy consumption per query in joules.

## Key Features

- **Energy-aware execution** — Real-time power monitoring, thermal throttling, per-query joule tracking, hardware-adaptive query plans
- **Hyperdimensional Computing** — HDC-encoded vectors for O(1) similarity search, 512-bit binary hypervectors with XOR binding
- **Multi-paradigm queries** — SQL, Cypher (graph), GraphQL, CQL (Cassandra), time-series (InfluxQL/PromQL)
- **Columnar analytics** — Arrow-based vectorized execution with zone-map pruning
- **5 transports** — HTTP REST, TCP binary protocol, WebSocket, WebTransport (HTTP/3 + QUIC), PostgreSQL wire protocol
- **4 client SDKs** — Rust (native), Python (PyO3), JavaScript/TypeScript (WASM), C (SQLite-style FFI)
- **Real-time subscriptions** — Pattern-based change notifications with HDC-based pre-filtering
- **21 domain verticals** — Finance, healthcare, gaming, geospatial, IoT, genomics, and more
- **Production-grade** — MVCC, RBAC, TLS encryption, 100+ Prometheus metrics, Kubernetes health checks, OpenTelemetry traces

## Quick Start

### From Source

```bash
git clone https://github.com/openIE-dev/jouledb.git
cd jouledb
cargo build --release

# Start the server (HTTP on :8080, TCP on :9000)
./target/release/joule-db-server
```

### Docker

```bash
docker build -t jouledb .
docker run -p 8080:8080 -p 9000:9000 -v ./data:/app/data jouledb
```

### CLI

```bash
cargo install --path joule-cli

# Server management
jouledb server start --foreground
jouledb server status
jouledb server stop

# Query
jouledb query execute "SELECT * FROM users WHERE age > 25"

# Interactive shell
jouledb shell
```

### PostgreSQL Wire Protocol

Connect with any PostgreSQL-compatible client:

```bash
psql -h localhost -p 5433 -U joule
```

## Architecture

### Five Concurrent Data Structures

Every ingested record is encoded into **five** structures at write time. The query engine picks the optimal structure for each operation automatically.

| Structure | Purpose | Query Type |
|-----------|---------|------------|
| HashMap | Record store | O(1) point lookups |
| B-Tree | Numeric/range index | Range scans, ORDER BY |
| String Index | Text search | Equality, prefix |
| HDC Hologram | Semantic similarity | O(1) approximate matching |
| Columnar (Arrow) | Analytics | Aggregations, OLAP |

**Write latency**: ~3-5μs per record. **Analytics speedup**: up to 2,678x vs row scan.

### Query Languages

| Language | Use Case |
|----------|----------|
| SQL | Relational queries |
| Cypher | Graph traversal |
| GraphQL | API queries |
| CQL | Wide-column (Cassandra) |
| InfluxQL / PromQL | Time-series analytics |

### Transports

| Transport | Port | Use Case |
|-----------|------|----------|
| HTTP REST | 8080 | General API access |
| TCP Binary | 9000 | High-throughput (16-byte header) |
| WebSocket | 8080 | Browser-compatible, real-time |
| WebTransport | — | Lowest-latency (HTTP/3 + QUIC) |
| PostgreSQL Wire | 5433 | psql-compatible, prepared statements |

## Energy Profiler

JouleDB is the first database to track energy consumption per query.

1. **Platform sensors** read power draw from hardware (macOS IOKit, Linux sysfs, or TDP-based estimation)
2. **Per-query tracking** measures joules consumed by each operation
3. **Hardware advisor** adjusts execution based on thermal state, memory pressure, and power budget
4. **Energy budgets** enforce per-query energy limits (e.g., `max_joules = 0.001`)

### Thermal States

| State | Action |
|-------|--------|
| `Nominal` | Full performance |
| `Fair` | Monitor |
| `Serious` | Reduce parallelism |
| `Critical` | Minimum operations |

```bash
# Energy status
curl http://localhost:8080/api/v1/energy
```

```json
{
  "power_watts": 12.4,
  "thermal_state": "Nominal",
  "cpu_temp_celsius": 62.3,
  "queries_tracked": 14892,
  "total_energy_joules": 847.23,
  "avg_joules_per_query": 0.0569,
  "advisor": {
    "recommendation": "full_performance",
    "throttle_level": 0
  }
}
```

## API Reference

### HTTP REST

```bash
# Health (Kubernetes-compatible)
curl http://localhost:8080/health
curl http://localhost:8080/health/live
curl http://localhost:8080/health/ready

# Key-value operations
curl -X POST http://localhost:8080/api/v1/keys/user:1 \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "age": 30}'

curl http://localhost:8080/api/v1/keys/user:1
curl -X DELETE http://localhost:8080/api/v1/keys/user:1

# Prometheus metrics
curl http://localhost:8080/metrics
```

### Real-Time Subscriptions

Subscribe to key changes using glob patterns over TCP, WebSocket, or WebTransport:

```json
{"type": "subscribe", "id": 1, "pattern": "users:*"}
{"type": "notification", "subscription_id": 42, "operation": "insert",
 "key": "users:1", "value": "{\"name\":\"Alice\"}"}
```

At 50+ active subscriptions, JouleDB automatically activates HDC-based pre-filtering.

## Client SDKs

### Rust

```rust
use joule_db_client::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::connect("127.0.0.1", 9000).await?;

    client.put("user:1", b"Alice", None).await?;
    let value = client.get("user:1").await?;

    let result = client.query("SELECT * FROM users WHERE age > 25", &[]).await?;
    let tx = client.begin().await?;
    tx.execute("INSERT INTO orders (id, total) VALUES (1, 99.99)", &[]).await?;
    tx.commit().await?;
    Ok(())
}
```

### Python

```python
from jouledb import Client

async def main():
    client = await Client.connect("127.0.0.1", 9000)
    await client.put("user:1", b"Alice")
    result = await client.query("SELECT * FROM users WHERE age > 25")
    await client.close()
```

### JavaScript / TypeScript (WASM)

```typescript
import { JouleDB } from '@jouledb/joule-db-js';

const db = await JouleDB.open({ backend: 'indexeddb' });       // Embedded
const db = await JouleDB.connect('http://localhost:8080');      // Remote

await db.put('user:1', '{"name": "Alice"}');
const result = await db.query<User>('SELECT * FROM users WHERE age > 25');

const subId = await db.subscribe('users:*', (event) => {
    console.log(`${event.operation}: ${event.key}`);
});
```

### C (FFI)

```c
#include "joule_db.h"

JouleDb *db;
joule_db_open("mydb.jdb", &db);

JouleStmt *stmt;
joule_db_prepare(db, "SELECT * FROM users WHERE id = ?", &stmt);
joule_db_bind_int(stmt, 1, 42);

while (joule_db_step(stmt) == JOULE_ROW) {
    const char *name = joule_db_column_text(stmt, 1);
}

joule_db_finalize(stmt);
joule_db_close(db);
```

## Domain-Specific HDC Modules

21 pre-built domain encoders that map industry-specific data types into hyperdimensional vectors:

| Domain | Crate | Use Cases |
|--------|-------|-----------|
| Finance | `joule-db-market-link` | Order books, trading strategies, risk |
| Healthcare | `joule-db-health-link` | Patient similarity, diagnosis patterns |
| Cybersecurity | `joule-db-cyber-link` | Threat detection, network anomalies |
| Gaming | `joule-db-gaming-link` | Matchmaking, cheat detection |
| Geospatial | `joule-db-spatial-link` | POI search, geofencing |
| IoT | `joule-db-iot-link` | Sensor fusion, anomaly detection |
| Genomics | `joule-db-genomics-link` | Sequence similarity, variant analysis |
| Supply Chain | `joule-db-supply-link` | Route optimization, demand forecasting |
| AdTech | `joule-db-adtech-link` | Audience targeting, campaign similarity |
| Legal | `joule-db-legal-link` | Case similarity, contract analysis |
| Energy | `joule-db-energy-link` | Grid optimization, consumption patterns |
| Telecom | `joule-db-telecom-link` | Network planning, traffic patterns |
| Insurance | `joule-db-insurance-link` | Claim similarity, fraud detection |
| Education | `joule-db-edu-link` | Learning paths, skill assessment |
| Agriculture | `joule-db-agri-link` | Yield prediction, disease detection |
| Automotive | `joule-db-auto-link` | Vehicle diagnostics, fleet management |
| Media | `joule-db-media-link` | Content recommendation |
| Retail | `joule-db-retail-link` | Product similarity, demand patterns |
| Graph | `joule-db-graph-link` | Relationship encoding, path similarity |
| Temporal | `joule-db-temporal-link` | Time-series patterns, event sequences |
| Multimodal | `joule-db-multimodal-link` | Cross-modal similarity |

## Benchmarks

```bash
cargo bench -p joule-db-benches --bench novel
```

| Operation | Performance |
|-----------|------------|
| Holographic KV get @ 100K keys | ~200K ops/sec |
| Holographic KV get @ 1M keys | ~200K ops/sec (O(1)) |
| Columnar SUM over 1M rows | < 1ms |
| Analytics vs row scan | 2,678x speedup |
| Zone-map pruning | 10x speedup |

## Project Structure

```
jouledb/
  joule-db-core/         # Storage engine, B-tree, MVCC, encryption, VFS
  joule-db-hdc/          # Hyperdimensional computing primitives
  joule-db-query/        # Multi-paradigm query engine (SQL, Cypher, GraphQL, CQL)
  joule-db-server/       # HTTP/TCP/WS/WebTransport server, auth, metrics
  joule-db-energy/       # Energy profiler, hardware advisor, thermal monitoring
  joule-db-gpu/          # GPU-accelerated operations
  joule-db-edge/         # Edge computing / IoT runtime
  joule-db-amorphic/     # Amorphic streaming data store
  joule-db-langgraph/    # LangGraph AI agent integration
  joule-cli/             # CLI administration tool
  joule-db-c/            # C FFI bindings (SQLite-style API)
  joule-db-js/           # JavaScript/TypeScript SDK (WASM)
  joule-db-domains/      # 21 domain-specific HDC modules
  joule-cloud/           # Cloud services (API gateway, billing, provisioner)
  clients/
    joule-db-client/     # Rust client SDK
    joule-db-python/     # Python SDK (PyO3)
    joule-quickstart/    # Quickstart examples
  benches/               # Performance benchmarks
  sigql/                 # SigQL query language
```

## Monitoring

- **Health**: `GET /health`, `/health/live`, `/health/ready` (Kubernetes-compatible)
- **Metrics**: `GET /metrics` (Prometheus, 100+ metrics)
- **Dashboard**: `GET /api/metrics` (JSON), `/api/metrics/history`, `/api/metrics/slow-queries`
- **Energy**: `GET /api/v1/energy` (power, thermal, per-query joules)
- **Observability**: OpenTelemetry traces, structured logging via `tracing`

## About

JouleDB is developed by [Open Interface Engineering](https://openie.dev). It is part of the [Joule ecosystem](https://joule-lang.org) — where energy is a first-class primitive across the language, database, and runtime.

- [jouledb.org](https://jouledb.org) — Product page
- [joule-lang.org](https://joule-lang.org) — The Joule programming language
- [projects.openie.dev](https://projects.openie.dev) — All openIE projects
- [blog.openie.dev](https://blog.openie.dev) — Engineering blog

## License

Licensed under MIT OR Apache-2.0.
