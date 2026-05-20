# JouleDB CLI Reference

**Version 1.0 — 2026-05-18**
**Binaries:** `joule` (`joule-cli`), `jouledb` (`joule-db-server` daemon), `joule-db-recover` (forensic), `scholar-recover` (Scholar-specific forensic)
**Sister docs:** [`MGAI-CLI-REFERENCE.md`](../MGAI-CLI-REFERENCE.md), [`MGAI-JWP-PROTOCOL.md`](../MGAI-JWP-PROTOCOL.md), [`MGAI-ACP-REFERENCE.md`](../MGAI-ACP-REFERENCE.md)

This is the JouleDB-specific CLI reference — separate from `MGAI-CLI-REFERENCE.md` which covers the user-facing `askdavidc` / `jmax` / `ask-server` binaries. The two are linked; this one is database-side.

---

## 1. `joule` — the universal CLI (`crates/bin/joule-cli`)

**Binary:** `joule`
**Crate:** [`crates/bin/joule-cli/`](../../crates/bin/joule-cli/)

One CLI across all five IDE brands (Lux, Joule, Arc, Data, DevOps). Every operation is metered in joules. JouleDB subcommands are under `joule db …`.

### 1.1 Global flags

| Flag | Env var | Default | Purpose |
|---|---|---|---|
| `-c <path>`, `--config <path>` | `JOULE_CONFIG` | — | Configuration file |
| `-o <fmt>`, `--output <fmt>` | — | `text` | Output format: `text`, `json`, `table` |
| `-v`, `--verbose` | — | off | Verbose output |
| `-q`, `--quiet` | — | off | Suppress non-essential output |
| `--version` | — | — | Print version and exit |

### 1.2 Database subcommands (`joule db …`)

| Subcommand | Purpose |
|---|---|
| `joule db server start` | Start a JouleDB server instance (wraps `joule-db-server`) |
| `joule db server stop` | Stop a running server |
| `joule db query shell` | Interactive SQL shell |
| `joule db query run <sql>` | One-shot query |
| `joule db ledger show` | Print the energy ledger for recent operations |
| `joule db ledger anchor` | Anchor pending receipts to the configured backend |
| `joule db recover` | Run forensic recovery — see §3 |

### 1.3 Non-database subcommands (for context)

| Subcommand | Purpose |
|---|---|
| `joule init <name>` | Create a session (defaults to `lux` brand) |
| `joule open <file>` | Open any `.{lux,joule,arc,data,devops}session` |
| `joule run <file>` | Execute source — routed through flowG |
| `joule check <path>` | Parse + typecheck |
| `joule build [--release]` | Compile via flowG pipeline |
| `joule dev` | Dev server with hot reload |
| `joule repl` | Interactive REPL |
| `joule search <query>` | verity-cascade search |
| `joule deploy` | Export flat files + deploy |
| `joule export <out>` | Export session to flat files |
| `joule ide` | Launch local IDE |
| `joule energy` | Show session energy report |
| `joule energy audit <path>` | Audit energy efficiency |

### 1.4 Environment variables

| Var | Purpose |
|---|---|
| `JOULE_CONFIG` | Override config file |
| `JOULE_DB_URL` | Default JouleDB connection URL (`jouledb://host:port`) |
| `ASKDAVIDC_MAX_ENERGY_UJ` | Energy budget cap per query (µJ) |
| `LOG_FORMAT` | `text` (default) or `json` |
| `RUST_LOG` | Standard tracing filter |

---

## 2. `jouledb` — the server daemon (`crates/joule-db-server`)

**Binary:** `jouledb`
**Crate:** [`crates/joule-db-server/`](../../crates/joule-db-server/)

The JouleDB server daemon. Direct invocation (without `joule db server start`) is supported for production deployments managed by systemd, OCI, or the `JouleDBCluster` K8s operator.

### 2.1 Synopsis

```bash
jouledb [--config <path>]
```

Configuration is supplied via a TOML file ([`config.rs`](../../crates/joule-db-server/src/config.rs)) or via env vars. Most operators run via systemd / K8s; the binary is one process per cluster replica.

### 2.2 Default ports

| Port | Protocol | Default | Purpose |
|---|---|---|---|
| 9000 | JWP/TCP | cloud | JouleDB wire protocol (managed cloud) |
| 9090 | JWP/TCP | local | JouleDB wire protocol (local default) |
| 9200 | JWP/TCP | test | `JwpServerConfig::default()` for testing |
| 5432 | pgwire/TCP | off until enabled | PostgreSQL-compatible wire |
| 8080 | HTTP | on | REST + health |
| (WS) | WebSocket | on | Real-time subscriptions |
| (WT) | WebTransport | off (`webtransport` feature) | HTTP/3 + QUIC |

### 2.3 Feature flags built into the binary

See [`crates/joule-db-server/README.md`](../../crates/joule-db-server/README.md) §"Feature flags" for the canonical list. Compile-time selection — the deployed binary's protocol surface is fixed at build.

### 2.4 systemd unit (example)

```ini
[Unit]
Description=JouleDB server (JWP)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/jouledb --config /etc/jouledb/config.toml
Environment=LOG_FORMAT=json JWP_PORT=9090 WS_PORT=9091
Restart=on-failure
RestartSec=5s
User=jouledb
Group=jouledb

[Install]
WantedBy=multi-user.target
```

### 2.5 Signal handling

| Signal | Action |
|---|---|
| `SIGTERM` | Graceful shutdown — drain JWP, flush WAL, close PVC |
| `SIGINT` | Same as `SIGTERM` |
| `SIGHUP` | Reload TOML config (for hot-tunable settings) |

---

## 3. `joule-db-recover` — forensic page-walker (`crates/joule-db-local/src/bin/`)

**Binary:** `joule-db-recover`
**Source:** [`crates/joule-db-local/src/bin/joule_db_recover.rs`](../../crates/joule-db-local/src/bin/joule_db_recover.rs)

### 3.1 When to run

`meta.wdb` is corrupted or points at the wrong tree, but `data.wdb` contains intact pages from the original B-tree. The engine refuses to open the database. You need to identify the real root and rewrite `meta.wdb` to point at it.

### 3.2 Synopsis

```bash
joule-db-recover <db_dir> \
    [--write-recovery-meta [--root <id>] |
     --write-synthetic-recovery |
     --rebuild-tree]
```

Three operating modes, mutually exclusive:

| Mode | What it does |
|---|---|
| default (scan) | Read-only: classifies pages, finds candidate roots, prints a report. Safe on a live database (CoW snapshot view). |
| `--write-recovery-meta [--root <id>]` | Atomically rewrites `meta.wdb` to point at the largest candidate root, or at the explicit `--root <id>`. Single-tree recovery. |
| `--write-synthetic-recovery` | Allocates a fresh page id, writes a synthetic B-tree internal page that points at the largest reachable subtree **per table prefix**, then rewrites `meta.wdb` to point at the synthetic page. Use when one corrupted root has eaten multiple tables. |
| `--rebuild-tree` | Most invasive. Sorts ALL leaf pages by first key, deduplicates overlapping ranges (skip-on-overlap, recent-wins tie-break), bulk-builds a fresh internal-node tree over the consistent leaf set, points `meta.wdb` at the new root. Use when no single existing root reaches the desired set of tables. |

### 3.3 Safety

- Each destructive mode preserves the previous `meta.wdb` as `meta.wdb.pre-recover-<unix_ts>`.
- The new `meta.wdb` goes through tmp + fsync + rename for atomicity.
- Read-only `scan` is safe to run against a live database — the JouleDB server's writes are CoW-additive, so the scan sees a consistent point-in-time view.

### 3.4 Memory and time

| Database size | Pages | Peak memory | Scan time |
|---|---|---|---|
| 64 GB | ~1M | ~40 MB | ~1 min |
| 594 GB | ~9M | ~360 MB | ~10 min |
| 1 TB | ~16M | ~640 MB | ~17 min |

(Sequential read at ~1 GB/s on NVMe; metadata cost ~50 bytes per page + ~12 bytes per internal-node child pointer.)

### 3.5 Validation of recovery

This binary was the tool that recovered Scholar's 594GB database in May 2026 after a `meta.wdb` corruption incident. The synthetic-recovery path was specifically designed for that case (multiple orphaned table-rooted subtrees, no single intact internal-node tree). See `the-fix-1.md` / `the-fix-2.md` (workspace root) for the incident narratives.

### 3.6 Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | Argument error |
| Other | I/O error or unrecoverable corruption (page count mismatch, magic-byte failure across many pages, etc.) |

---

## 4. `scholar-recover` — Scholar-specific forensic recovery (`crates/bin/scholar-recover`)

**Binary:** `scholar-recover`
**Crate:** [`crates/bin/scholar-recover/`](../../crates/bin/scholar-recover/)

A higher-level forensic tool layered on top of `joule-db-recover`. While `joule-db-recover` rebuilds the JouleDB internal-node tree, `scholar-recover` extracts row data from leaf pages and re-ingests it into a fresh JouleDB. Used when the corruption is severe enough that even synthetic-recovery can't reconstruct a consistent index.

### 4.1 When to run

- The catalog B-tree references pages that don't exist (dangling-pointer allocator bug).
- WAL is empty so no replay helps.
- A clean ingest restart would cost more than the recovery time (Scholar's ingest is 48-100 h).

The 2026-04-21 Scholar incident hit exactly this scenario.

### 4.2 Synopsis

```bash
scholar-recover scan    --src /path/to/corrupt/data.wdb
scholar-recover rebuild --src /path/to/corrupt/data.wdb --dest /fresh/db/dir
```

### 4.3 Subcommands

| Subcommand | Purpose |
|---|---|
| `scan` | Linear scan — counts pages by type, counts recoverable rows per table. Read-only. |
| `rebuild` | Scan + emit a fresh JouleDB at `--dest`. |

### 4.4 `rebuild` options

| Flag | Default | Purpose |
|---|---|---|
| `--src <path>` | (required) | Corrupt `data.wdb` to recover from |
| `--dest <dir>` | (required) | Fresh JouleDB directory to write to |
| `--batch-size <n>` | (sane default) | Rows per write transaction |
| `--max-rows <n>` | 0 (unlimited) | Hard cap on extracted rows |

### 4.5 Strategy

`scholar-recover` bypasses the B-tree entirely. It treats `data.wdb` as an opaque sequence of 64 KB pages and for each page:

1. Decodes the 32-byte header (magic `0x57444250`, page type, `data_len`).
2. If `page_type == BTreeLeaf` and `data_len > 0`, deserializes the body as a B-tree leaf — recovering `(key, value)` pairs.
3. For inline values, emits them directly. For overflow-marker values, follows the page-chain by ID and reassembles; drops the entry on any break.
4. Keys matching `row::{table}\x00` get routed back into a fresh JouleDB under the same table name, using the clean `joule_db_local::Database` write path.

Missing a handful of leaf pages just loses those specific rows; the rest survive. B-tree internal pages carry no row data, so their corruption is harmless to this tool.

### 4.6 Choice between `joule-db-recover` and `scholar-recover`

| Symptom | Tool |
|---|---|
| `meta.wdb` corrupted, original tree intact | `joule-db-recover --write-recovery-meta` |
| Multiple table-rooted subtrees, no single intact root | `joule-db-recover --write-synthetic-recovery` |
| Internal-node tree fully corrupted, leaves intact | `joule-db-recover --rebuild-tree` |
| Catalog B-tree itself references dangling pages, WAL empty | `scholar-recover rebuild` |
| Severe corruption, want a clean DB and willing to drop indexes | `scholar-recover rebuild` |

When in doubt, run the read-only `scan` mode of both tools first and compare reports.

---

## 5. ACP context — JouleDB binaries as agent processes

The OpenIE ACP (Agent CLI / Container Protocol) contract from [`MGAI-ACP-REFERENCE.md`](../MGAI-ACP-REFERENCE.md) applies to JouleDB binaries with the following specifics:

### 5.1 Invocation class

| Binary | Class | I/O model | Lifetime |
|---|---|---|---|
| `joule` (subcommands like `joule db query run`) | One-shot | stdin → stdout, exit on completion | seconds |
| `joule db query shell` | REPL | Line-delimited stdio | until EOF |
| `jouledb` | Daemon | TCP (JWP 9000/9090) + WebSocket | indefinite, signal-managed |
| `joule-db-recover scan` | One-shot | stdin → stdout | minutes |
| `joule-db-recover --write-recovery-meta` | One-shot, destructive | stdin → stdout | minutes |
| `scholar-recover scan` | One-shot | stdin → stdout | minutes |
| `scholar-recover rebuild` | One-shot, write-out | stdin → stdout | hours |

### 5.2 Container packaging

The `jouledb` daemon is packaged as a Kubernetes `StatefulSet` workload via the `JouleDBCluster` CRD (`jouledb.cloud/v1`). See [`docs/jouledb/CLOUD-OPERATOR.md`](CLOUD-OPERATOR.md) for the operator runbook.

`joule-db-recover` is **not** packaged as a separate workload — it's compiled into the same image as `jouledb` so it can `kubectl exec` against a stuck pod for in-place recovery:

```bash
kubectl exec -it my-cluster-0 -n jouledb -- \
  joule-db-recover scan /data/joule.db
```

### 5.3 stdio separation (per ACP §2.3)

| Stream | Reserved for | Format |
|---|---|---|
| **stdout** | Query result / status report | UTF-8, line-delimited where applicable; JSON if `--output json` |
| **stderr** | Logs, diagnostics, progress | Tracing output (text or JSON per `LOG_FORMAT`) |
| **stdin** | Query / REPL input | UTF-8 |

### 5.4 Exit codes (per ACP)

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Generic failure |
| 2 | Argument parsing error (clap convention) |
| Others | Tool-specific I/O / corruption / energy-budget errors |

---

## 6. See also

- [`MGAI-CLI-REFERENCE.md`](../MGAI-CLI-REFERENCE.md) — `askdavidc` / `jmax` / `ask-server` / `ask-mcp` (user-facing surface)
- [`MGAI-ACP-REFERENCE.md`](../MGAI-ACP-REFERENCE.md) — process contract for all OpenIE agent binaries
- [`MGAI-JWP-PROTOCOL.md`](../MGAI-JWP-PROTOCOL.md) — wire protocol the `jouledb` daemon speaks
- [`docs/jouledb/CLOUD-OPERATOR.md`](CLOUD-OPERATOR.md) — operator runbook for managed deployments
- [`crates/joule-db-server/README.md`](../../crates/joule-db-server/README.md) — daemon module map
- [`the-fix-1.md`](../../the-fix-1.md), [`the-fix-2.md`](../../the-fix-2.md) — Scholar recovery incident narratives

---

*Drafted 2026-05-18 as wave 3 of the JouleDB documentation parity pass.*
