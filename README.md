<p align="center">
  <img src="https://jouledb.org/favicon.svg" width="80" alt="JouleDB" />
</p>

<h1 align="center">JouleDB</h1>

<p align="center">
  <strong>The world's first energy-aware database</strong> — powered by Hyperdimensional Computing, hardware-adaptive, with real-time J/query telemetry.
</p>

<p align="center">
  <a href="https://jouledb.org">Website</a> · <a href="https://projects.openie.dev">Projects</a> · <a href="https://blog.openie.dev">Blog</a> · <a href="https://openie.dev">Open Interface Engineering</a>
</p>

---

```
       ┌──────────────────────────────────────────────────────┐
       │                 jouledb run <engine>                  │
       │          postgres · mysql · redis · mongodb           │
       └──────────────────────┬───────────────────────────────┘
                              │
       ┌──────────────────────▼───────────────────────────────┐
       │                   jouled daemon                       │
       │       Health Monitor · Energy Dashboard (:7000)       │
       │            Unix Socket IPC · Auto-Restart             │
       └──────────────────────┬───────────────────────────────┘
                              │
       ┌──────────────────────▼───────────────────────────────┐
       │                  JouleDB Engine                       │
       │                                                       │
       │  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌───────────┐ │
       │  │ HashMap  │ │ B-Tree  │ │ HDC Holo │ │ Columnar  │ │
       │  │  O(1)    │ │  Range  │ │  O(1)    │ │  Arrow    │ │
       │  └─────────┘ └─────────┘ └──────────┘ └───────────┘ │
       │                                                       │
       │  ┌──────────────────────────────────────────────────┐ │
       │  │ Energy Monitor · Platform Sensors · J/query      │ │
       │  └──────────────────────────────────────────────────┘ │
       └──────────────────────────────────────────────────────┘
```

JouleDB is a hybrid OLTP/OLAP/semantic database written in **Rust** (Edition 2024, 60+ crate workspace). It encodes every record into five concurrent data structures at write time, adapts execution based on hardware power/thermal state, and tracks energy consumption per query in joules.

**3,786 tests. Zero failures. 3-node Raft cluster. 11 adversarial stress test suites.**

## Install

```bash
curl -fsSL https://jouledb.org/install.sh | sh
```

Or download binaries from [Releases](https://github.com/openIE-dev/jouledb/releases).

## Quick Start

```bash
# Start the persistent daemon
jouledb-cli daemon start
# → Dashboard: http://127.0.0.1:7000

# Run any database with energy telemetry
jouledb-cli run postgres
jouledb-cli run redis --port 6380
jouledb-cli run mysql

# Or run JouleDB natively
jouledb-cli server start --foreground
jouledb-cli query execute "SELECT * FROM users WHERE age > 25"
jouledb-cli shell

# Connect with psql (PostgreSQL wire protocol)
psql -h localhost -p 5433 -U joule

# Auto-start on login (macOS launchd / Linux systemd)
jouledb-cli daemon install
```

## Universal Database Energy Layer

Run **any database** with Joule energy telemetry. One command. Automatic energy sidecar. No code changes.

```bash
$ jouledb-cli run postgres

  PostgreSQL  5432  native
  Energy   http://127.0.0.1:15432/energy
  Data     ./postgres-data-a1b2c3d4

  Apple M4 Max  92W  GPU + Neural Engine

$ curl localhost:15432/energy
{
  "engine": "postgres",
  "power_watts": 8.2,
  "total_energy_joules": 1247.5
}
```

Supported engines: **PostgreSQL**, **MySQL**, **Redis**, **MongoDB**, **SQLite**, and **JouleDB** itself.

## Persistent Daemon

A background daemon manages all database instances — Docker Desktop parity for databases.

- **Instance management** — start, stop, list across sessions via Unix socket IPC
- **Health monitor** — automatic health checks every 10s, auto-restart on crash (max 3)
- **Energy dashboard** — unified HTTP dashboard on port 7000 (JSON API + Prometheus + HTML)
- **Orphan recovery** — detects dead PIDs on startup, marks as failed
- **OS service** — `jouledb-cli daemon install` generates macOS launchd or Linux systemd service files

```bash
jouledb-cli daemon start       # Start daemon
jouledb-cli daemon status      # Show uptime, instances, energy
jouledb-cli daemon stop        # Graceful shutdown
jouledb-cli daemon install     # Auto-start on login
```

## Key Features

### Five Concurrent Data Structures

Every ingested record is encoded into five structures at write time. The query engine picks the optimal structure automatically.

| Structure | Purpose | Performance |
|-----------|---------|-------------|
| HashMap | Record store | O(1) point lookups |
| B-Tree | Range index | Range scans, ORDER BY |
| String Index | Text search | Equality, prefix, full-text |
| HDC Hologram | Semantic similarity | O(1) approximate matching |
| Columnar (Arrow) | Analytics | 2,678x vs row scan |

**Write latency**: ~3-5μs per record.

### Energy-Aware Execution

- **Platform sensors** — macOS IOKit, Linux sysfs, or TDP-based estimation
- **Per-query joule tracking** — every query reports energy consumed
- **Thermal-adaptive** — four states (Nominal, Fair, Serious, Critical) drive automatic throttling
- **Hardware advisor** — adjusts execution based on thermal state, memory pressure, power budget
- **Energy budgets** — enforce per-query limits (`max_joules = 0.001`)

### Distributed Replication (HRP)

Holographic Replication Protocol — Raft consensus with superpowers:

- **Event-driven replication** — Notify-based, not polling
- **Mutation deltas** — INSERT/CREATE/DROP captured as binary deltas, followers bypass SQL parsing
- **Write tokens** — HMAC-SHA256 with HKDF epoch key derivation, replay prevention
- **Erasure coding** — Reed-Solomon GF(2^8), 2 data + 1 parity shard
- **Energy-aware routing** — route reads to lowest-energy peer
- **Binary wire protocol** — HRP v2 envelope with CRC32
- **TLS/mTLS** — encrypted inter-node communication

| Metric | Baseline | HRP | Change |
|--------|----------|-----|--------|
| DDL latency | 117ms | 59.5ms | **-49%** |
| INSERT avg | 116ms | 67.8ms | **-42%** |
| Parallel writes | 37.6 w/s | 76.3 w/s | **+103%** |
| Replication lag | 348ms | 216ms | **-38%** |
| Cluster reads | 228 r/s | 309 r/s | **+35%** |

*Measured on 3-node Apple Silicon Thunderbolt cluster (M4 Max + 2x M3 Ultra)*

### Runtime Isolation

Run databases in native, VM, or WASM sandboxes:

```bash
jouledb-cli server start --mode native    # Direct process (default)
jouledb-cli server start --mode vm        # VM-isolated via InvisibleVM
jouledb-cli server start --mode wasm      # WASM sandbox
```

### Multi-Paradigm

| Feature | Details |
|---------|---------|
| **7 Query Languages** | SQL, Cypher, GraphQL, CQL, InfluxQL/PromQL, SigQL, Feature Store |
| **5 Transports** | HTTP REST, TCP binary (16-byte header), WebSocket, WebTransport (HTTP/3 + QUIC), PostgreSQL wire |
| **15 Client SDKs + ODBC** | Rust, Python, JS/TS, C FFI, Go, Java (+ JDBC), C#/.NET, Swift, Kotlin, Ruby, PHP, Zig, Dart/Flutter, Elixir, no_std Rust, ODBC |
| **21 Domain Modules** | Finance, healthcare, cybersecurity, genomics, IoT, supply chain, adtech, legal, energy, telecom, and more |
| **Cloud Services** | API gateway, control plane, billing, provisioner with persistence |

### Production-Grade

- **Auth**: RBAC, SCRAM-SHA-256, TLS/mTLS (rustls)
- **MVCC**: Snapshot isolation transactions
- **Observability**: 100+ Prometheus metrics, OpenTelemetry traces, Grafana dashboards
- **Kubernetes**: Health, liveness, readiness probes
- **Security**: Fuzz testing, injection testing, dependency scanning, SAST, secrets detection
- **Testing**: 3,786 tests, 11 adversarial stress test suites (413 stress tests)

## Benchmarks

| Operation | Performance |
|-----------|------------|
| Holographic KV get @ 1M keys | ~200K ops/sec (O(1)) |
| Columnar SUM over 1M rows | < 1ms |
| Analytics vs row scan | 2,678x speedup |
| Zone-map pruning | 10x speedup |
| HRP parallel writes | +103% throughput |

```bash
cargo bench -p joule-db-benches --bench novel
```

## API

```bash
# Energy status
curl http://localhost:8080/api/v1/energy

# Health (Kubernetes-compatible)
curl http://localhost:8080/health

# Key-value
curl -X POST http://localhost:8080/api/v1/keys/user:1 \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "age": 30}'

# Prometheus metrics
curl http://localhost:8080/metrics

# Daemon energy dashboard
curl http://localhost:7000/api/energy
curl http://localhost:7000/api/instances
curl http://localhost:7000/metrics
```

## About

JouleDB is developed by [Open Interface Engineering](https://openie.dev). It is part of the [Joule ecosystem](https://joule-lang.org) — where energy is a first-class primitive across the language, database, and runtime.

- [jouledb.org](https://jouledb.org) — Product page
- [joule-lang.org](https://joule-lang.org) — The Joule programming language
- [projects.openie.dev](https://projects.openie.dev) — All openIE projects
- [blog.openie.dev](https://blog.openie.dev) — Engineering blog

## License

Licensed under MIT OR Apache-2.0.
