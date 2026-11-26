# JouleDB (`v0.9.0` — Pre-1.0 Research Preview)

**An experimental energy-metered multi-model database in Rust (`Apache-2.0 OR MIT`) exploring Copy-on-Write `SSI` storage, out-of-core `.jcol` vectorized columnar analytics (`43 / 43` Official ClickBench at `100M` rows), PostgreSQL `pgwire` SQL, multi-host WAN TCP Raft `ReadIndex` consensus, Claude/MCP Apps Studio UI (`SEP-1865`), `10,240`-bit `Amorphic` VSA storage, and `.jsyn` sub-1-bit sentence embeddings.**

[![Release](https://github.com/openIE-dev/jouledb/actions/workflows/release.yml/badge.svg)](https://github.com/openIE-dev/jouledb/releases)
[![Quality](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml/badge.svg)](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml)
[![Docs](https://github.com/openIE-dev/jouledb/actions/workflows/docs.yml/badge.svg)](https://openie-dev.github.io/jouledb)
[![crates.io](https://img.shields.io/crates/v/jouledb.svg)](https://crates.io/crates/jouledb)
[![Hugging Face Embedder](https://img.shields.io/badge/Hugging%20Face-MiniLM%20.jsyn%20(cos%200.944)-0D9488)](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)
[![Website](https://img.shields.io/badge/Web-jouledb.org-2563EB)](https://jouledb.org)

> **Pre-1.0 Status & `v0.9.0` Release Summary (`2026-11-26`):** Building on our `v0.6.0` external-audit remediation (`v1.0.0` GA tag retracted/yanked, unrelated `1.11M`-LOC `joule-web` crate and synthetic `frontier_exam_routes.rs` deleted, 81 unverified low-BPW causal LLM exports pruned from Hugging Face) and `v0.7.0`–`v0.8.0` storage/analytical upgrades, **`v0.9.0`** ships native **Claude/MCP Apps (`SEP-1865`)**, **1.72× faster full-`100M` ClickBench**, **3.0× faster `pgwire` Extended Pipeline OLTP**, and **multi-host WAN Raft `elle` verification**:
> - **Claude Apps & MCP Apps (`SEP-1865` Interactive Studio + `.mcpb` Bundles):** Built-in **JouleDB Studio (`ui://jouledb/studio.html`, `text/html;profile=mcp-app`)** served directly from the `jouledb` binary over `jouledb mcp --data <dir>` (`stdio`) and `POST /mcp` (`50` tools incl. `db.studio`), plus one-click **Claude Desktop `.mcpb` (`manifest_version: "0.3"`)** bundles and an Antigravity plugin (`deploy/antigravity-plugin/`).
> - **Full `99,997,497`-Row (`14.78 GB`) Official ClickBench Accelerated `1.72×` (`4.85× -> 2.82×` Geomean vs DuckDB `2.0.1`, `15.48 GB` Peak `VmHWM`):** Added zero-I/O segment `global_int_sum` metadata (`Q2, Q3, Q29`), dynamic `AtomicI64` Top-K ZoneMap bound pruning (`Q19, Q23, Q24, Q26`), zero-`Arc<str>` `RawTextSlice` text distinct/grouping (`Q5, Q12, Q33, Q34`), and 256-partition `CompactI2Row` in-place sort-reduce — cutting `100M`-group `Q32` **7.24×** (`17.42 s -> 2.41 s`), lowering `100M` peak `VmHWM` **5.26 GB** (`20.74 GB -> 15.48 GB`), cutting `100M` geomean `reported_j` **1.75×** (`4.25× -> 2.43×`), and beating DuckDB `2.0.1` on **`8 / 43` queries at `100M` rows** (`13 / 43` at `1M`, `12 / 43` at `10M`, `21 / 43` at `100k`).
> - **`pgwire` Extended Query Static `Describe` & `100`-Warehouse OLTP (`3.01×` Faster Pipelined TPS):** Added static AST/catalog `describe_sql_static` OID inference and `DROP TABLE` index metadata cleanup — raising `psycopg` Extended Pipeline throughput **3.01×** (`593.7 -> 1,790.0 TPS` at `c=8`, `1,703.3 TPS` at `c=16`; **`3,351.0 TPS`** Simple Query at `c=16`, `invariant_verified = true` across all 5 engine modes vs `PostgreSQL 16.16` & `SQLite 3.53.1`).
> - **5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft + Official Clojure `elle 0.2.8` (`{:valid? true}`):** Executed 5 real OS `raft_wan_node` processes across **2 distinct physical Linux hosts** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM, `27.58 ms` median WAN TCP RTT) with cross-WAN `2/5` leader isolation, real OS `SIGKILL` (`kill -9`), and `raft.wal` crash recovery -> **`{:valid? true}`** (`270` events, `:strict-serializable`).
> - **Prebuilt Python Wheels & TypeScript Tarball on GitHub Releases:** Attached `cp39-abi3` `.whl` wheels (macOS ARM64 & Linux `manylinux_2_34_x86_64`), `openie-jouledb-0.9.0.tgz`, and `.mcpb` bundles to GitHub Release `v0.9.0` (not yet published to PyPI/npm/NuGet). See [`CHANGELOG.md`](./CHANGELOG.md).

---

## What Is Real & Verified Today (`41` Workspace Crates · `735.2k` LOC · `12,272` Tests)

1. **Core Storage, MVCC `SSI` & Recovery (`joule-db-core`, `joule-db-local`):**
   - Copy-on-Write B-tree pages, lock-free positional `pread`/`pwrite`, Cahill Serializable Snapshot Isolation (`SSI` `rw`-antidependency detection), index-assisted `MvccTableStorage` with `O(1)` PK/UNIQUE index checks and group-committed WAL durability, `CRC-32` torn-WAL truncation, `ALP + FSST` columnar codecs, and `RaBitQ` 1-bit `POPCNT` vector quantization.
2. **Out-of-Core `.jcol` Columnar Analytics, `pgwire` & Incremental Views (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire listener (`psql`, `psycopg3` Simple & static-`Describe` Extended Pipeline modes), row-budgeted streaming Parquet `COPY` spilling `262,144`-row `.jcol` disk segments (`I8/I16/I32/I64` narrowing, `global_int_sum` + ZoneMap headers, compact `u8/u16/u32` local dictionaries, CRC-32 integrity), two-phase `WHERE`-first `pread` segment execution + 256-partition `CompactI2Row` / Radix-64 `GROUP BY`, and `DBSP` `ZSet<T>` incremental view algebra.
3. **Claude Apps / MCP Apps Host (`mcp_app_ui.rs`, `mcp_transport.rs`):**
   - Native `SEP-1865` MCP Apps server over `jouledb mcp --data <dir>` (`stdio` JSON-RPC `2025-06-18`) and `POST /mcp` (Streamable HTTP), exposing `50` tools (`db.studio`, `db.query`, `db.schema`, `db.amorphic.*`, `db.synapse.*`, `db.branch.*`, `db.ledger.*`) and a zero-CDN interactive **JouleDB Studio** (`ui://jouledb/studio.html`, `text/html;profile=mcp-app`) for Claude Desktop, Claude Web, VS Code, Goose, and Antigravity.
4. **5-Node Multi-Host WAN & Local TCP Raft Consensus (`raft.rs`, `raft_transport.rs`, `examples/raft_wan_node.rs`):**
   - Quorum write commit via `RaftNode::propose_and_commit` (§5.4.2), quorum linearizable reads via `RaftNode::read_linearizable` (§6.4 `ReadIndex` heartbeat quorum, zero WAL growth), `CRC-32` `raft.wal` crash recovery, and both multi-host Tailscale WAN (`27.58 ms` RTT + `SIGKILL`) and local 20-link fault-injection suites verified **`{:valid? true}`** by official Clojure `elle/elle 0.2.8`.
5. **Energy Telemetry & `.jsyn` Sub-1-Bit Sentence Embeddings (`joule-db-energy`, `joule-db-weights`):**
   - Bare-metal hardware counter readers (**`reported_j`** via Linux `intel-rapl` and macOS `IOKit SMC`) segregated from analytical cost models (**`est_j`**), plus pure-Rust [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`1.95 MB`, `0.4378 BPW`, **`0.9441` cosine** vs `FP32`).

---

## Reproducible External Evaluations (`bench/`)

### 1. Full Official ClickBench (`hits.parquet`, `99,997,497` Rows, `105` Columns, `43` Queries) vs. DuckDB `2.0.1`

Executed via [`bench/clickbench_official.py`](./bench/clickbench_official.py) over real `pgwire` (`psycopg 3.3.9`) on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, `128 GB` RAM, Linux `rapl-package` **`reported_j`**):

| Official `hits.parquet` Scale | Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB Load RSS (`VmHWM` Peak) | Queries Succeeded | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`100,000` rows** ([JSON](./bench/results/clickbench_official_100k.json)) | `0.326 s` vs `0.409 s` | `927 MB` (`2,719 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`35` exact) | **`21`** / `22` | **`1.31×` slower** (`median 1.04×`) | **`1.65×` more J** |
| **`1,000,000` rows** ([JSON](./bench/results/clickbench_official_1m.json)) | `0.257 s` vs `1.363 s` | `1,246 MB` (`1,959 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`13`** / `30` | **`2.42×` slower** (`median 3.22×`) | **`2.34×` more J** |
| **`10,000,000` rows** ([JSON](./bench/results/clickbench_official_10m.json)) | `1.459 s` vs `12.568 s` | `2,163 MB` (`9,060 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`12`** / `31` | **`2.52×` slower** (`median 3.88×`) | **`2.47×` more J** |
| **`99,997,497` rows — FULL `14.78 GB`** ([JSON](./bench/results/clickbench_official_100m.json)) | `16.497 s` vs `86.068 s` | **`6,204 MB` (`15,483 MB`)** | **`43 / 43`** (`0` err) | **`43 / 43`** (`31` exact) | **`8`** / `35` | **`2.82×` slower** (`median 3.42×`) | **`2.43×` more J** |

- **Candid Full-`100M` Summary:** In `v0.9.0`, out-of-core `.jcol` segment execution (`6.20 GB` post-load RSS, **`15.48 GB` kernel `VmHWM` peak** — down `5.26 GB` from `v0.8.0`) narrows the full `100M`-row geomean latency gap vs DuckDB `2.0.1` from **`4.85×` to `2.82×`** (`2.43×` `reported_j`), cuts `100M`-group `Q32` **7.24×** (`17.42 s -> 2.41 s` vs DuckDB `1.06 s`), and beats DuckDB `2.0.1` on **`8 / 43` queries at `100M` rows** (`Q0` `1.51 ms` vs `9.36 ms`, `Q2` `1.60 ms` vs `8.93 ms`, `Q3` `1.84 ms` vs `12.77 ms`, `Q6` `1.24 ms` vs `7.33 ms`, `Q19` `31.32 ms` vs `42.46 ms`, `Q28` `2,912 ms` vs `3,443 ms`, `Q29` `2.27 ms` vs `13.68 ms`, `Q42` `12.00 ms` vs `23.70 ms`). DuckDB `2.0.1` wins **`35 / 43` queries (`2.82×` geomean, `5.22×` faster Parquet load)**, with JouleDB's widest ratio on `CASE WHEN`-grouped `Q39` (`519.3 ms` vs `23.2 ms`, `22.4×`).

### 2. Head-to-Head `100`-Warehouse TCP `pgwire` OLTP vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

Executed via [`bench/oltp_pgwire.py`](./bench/oltp_pgwire.py) ([`oltp_pgwire_100wh.json`](./bench/results/oltp_pgwire_100wh.json), `rapl-package` **`reported_j`**): `100,000` accounts (`100` warehouses), `50%` indexed point read, `30%` durable `BEGIN; UPDATE accounts ...; INSERT INTO history ...; COMMIT` (`fsync=on`, `synchronous_commit=on`), `20%` warehouse `COUNT`/`SUM` across `1, 4, 8, 16` concurrent TCP clients:

| Engine & Wire Protocol (`invariant_verified = true` on all 5) | `c = 1` TPS (`p50`) | `c = 4` TPS (`p50`) | `c = 8` TPS (`p50`) | `c = 16` TPS (`p50`, `mJ/txn`) | Committed Writes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **JouleDB `0.9.0` (`pgwire` Simple Query, 4-RTT + `fsync`)** | `522.2` (`0.34 ms`) | `954.5` (`0.97 ms`) | `2,125.1` (`1.04 ms`) | **`3,351.0`** (`1.07 ms`, `24.3 mJ`) | `16,647` (`0` errors) |
| **JouleDB `0.9.0` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `260.0` (`0.57 ms`) | `871.9` (`1.58 ms`) | **`1,790.0`** (`2.07 ms`) | **`1,703.3`** (`2.09 ms`, `75.7 mJ`) | `10,973` (`1` SI conflict) |
| **PostgreSQL `16.16` (`pgwire` Simple Query, 4-RTT + `fsync`)** | `3,505.0` (`0.17 ms`) | `2,957.2` (`0.21 ms`) | `6,100.7` (`0.12 ms`) | **`10,815.9`** (`0.38 ms`, `9.13 mJ`) | `55,902` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `1,291.4` (`0.08 ms`) | `2,666.5` (`0.20 ms`) | `5,607.2` (`0.30 ms`) | **`8,634.4`** (`0.68 ms`, `14.7 mJ`) | `43,602` (`0` conflicts) |
| **SQLite `3.53.1` (Embedded in-proc C API, `WAL`, `NORMAL`)** | `33,431.1` (`0.006 ms`) | `56,406.6` (`0.008 ms`) | `55,520.2` (`0.011 ms`) | **`60,263.4`** (`0.012 ms`, `2.18 mJ`) | `506,957` (no TCP / `fsync`) |

- **Candid OLTP Summary:** Static AST/catalog `describe_sql_static` OID inference + `DROP TABLE` index metadata cleanup raised JouleDB Extended Query `pipeline()` throughput **3.01×** (`593.7 -> 1,790.0 TPS` at `c=8`, `1,703.3 TPS` at `c=16`, `2.50×` more committed writes), while Simple Query reaches **`3,351.0 TPS`** at `c=16` (`0` errors, `invariant_verified = true`). `PostgreSQL 16.16` remains **`3.23×` faster** at `c=16` (`10,815.9 TPS`) due to shared-buffer page-level locking vs JouleDB's table-wide `commit_lock`.

### 3. 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

Executed via [`bench/raft_multihost_wan.py`](./bench/raft_multihost_wan.py), [`examples/raft_wan_node.rs`](./crates/joule-db-server/examples/raft_wan_node.rs), and [`bench/elle_check.clj`](./bench/elle_check.clj):
- **Cross-Machine WAN Topology:** 5 real OS `raft_wan_node` processes with durable `raft.wal` across **2 distinct physical Linux hosts** (`framework-128gb-0000` bare-metal `n1..n3` + `openie-core-1` Cloud Debian 12 VM `n4..n5`) over encrypted Tailscale WAN (**`27.58 ms` median TCP RTT**, `p95 28.65 ms`).
- **Nemeses:** Cross-WAN `2/5` minority split (`["n1", "n4"]` vs `["n2", "n3", "n5"]`) isolating initial leader `n1` -> stale write & `ReadIndex` read rejected -> `3/5` majority elects `n3` (term `2`, `24` committed ops) -> real OS **`SIGKILL` (`kill -9`)** on cloud node `n5` (`16` ops committed during crash) -> `raft.wal` restart & 5-node convergence (`commit=68, applied=68, x=68, y=39` across all 5 nodes).
- **Official `elle/elle 0.2.8` Verdict ([`raft_wan_multihost_elle_verdict.txt`](./bench/results/raft_wan_multihost_elle_verdict.txt)):** **`{:valid? true}`** (`rw-register`, `:strict-serializable`, `270` WAN events; local 20-link `tests/raft_tcp_elle_fault_injection.rs` also `{:valid? true}`).

### 4. `.jsyn` (`JSYN-v1`) Fidelity vs. Compression Curve

| Model / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Guidance |
| :--- | :---: | :---: | :---: | :--- |
| **[`all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.438` | `22.2×` (`1.95 MB`) | **`0.9441` embedding cosine** vs `FP32` | **Recommended** for semantic search & `TriModal RRF` |
| `Gemma-4-E2B` `q_proj` (`r = 8..128`) | `0.028–0.166` | `564×–96.6×` | `0.271–0.455` output cos (`KL 3.87–2.72`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE, `r = 8..128`) | `0.023–0.134` | `701×–120×` | `0.113–0.350` output cos (`KL 2.91–2.59`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |

---

## Quickstart (`v0.9.0`)

```bash
# 1. Install prebuilt v0.9.0 binary (macOS ARM64/x86_64; Linux x86_64 GLIBC <= 2.35, Ubuntu 22.04/24.04 & Debian 12 ready)
cargo binstall jouledb

# 2. Launch JouleDB server (HTTP :8080, TCP :8081, PgWire :8082, JWP :8083, FlightSQL :8084, MCP HTTP /mcp)
jouledb --port 8080 --data ./joule-data

# 3. Or run as a Claude Desktop / MCP Apps stdio server (50 tools + interactive ui://jouledb/studio.html)
jouledb mcp --data ./joule-data

# 4. Connect with psql, install GitHub Release Python/TS SDK artifacts, or run smoke/security self-tests
psql -h 127.0.0.1 -p 8082 -U postgres -c "SELECT version();"
jouledb self-check && jouledb trust
```

---

## Honest Engineering Boundaries (Read Before Adopting)

- **Pre-1.0 research preview (`v0.9.0`), not a battle-tested production RDBMS:** While JouleDB `0.9.0` executes all `43 / 43` Official ClickBench queries on `99,997,497` rows (`2.82×` geomean vs DuckDB `2.0.1`, `15.48 GB` peak `VmHWM`), `3,351 TPS` Simple / `1,790 TPS` Pipelined `pgwire` OLTP, and 5-node cross-machine WAN Raft (`elle` `{:valid? true}`), mature specialized engines (`PostgreSQL 16` at `10.8k TPS` `c=16`, `DuckDB 2.0` at `2.82×` faster `100M` geomean, `SQLite` embedded WAL) remain faster for conventional single-model production workloads.
- **No external compliance certifications:** `jouledb trust` exercises technical security primitives in code (`external_audit_performed = false`). **No third-party SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been conducted.**
- **Exact compiler safety lint scope:** `#![deny(unsafe_code)]` is set in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`). `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` is enforced in **7** core crates (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`, `joule-db-local`, `joule-db-energy`, `joule-db-weights`, `joule-db-query`, plus `#![deny(clippy::unwrap_used)]` on `disk_col_segments.rs` and `mcp_app_ui.rs`); the workspace-wide lint level is `warn`.
- **SDK registry status:** Prebuilt `cp39-abi3` Python `.whl` wheels, TypeScript `openie-jouledb-0.9.0.tgz`, and Claude `.mcpb` bundles are published on [GitHub Releases (`v0.9.0`)](https://github.com/openIE-dev/jouledb/releases/tag/v0.9.0); packages are **not yet published** to PyPI, npm, NuGet, or Maven Central.

---

## Documentation & Links

- **Website:** [https://jouledb.org](https://jouledb.org) ([Claude & MCP Apps](https://jouledb.org/claude) · [Benchmarks](https://jouledb.org/benchmarks) · [Honest Boundaries](https://jouledb.org/honest) · [Security Controls & Bridges](https://jouledb.org/trust) · [`.jsyn` Fidelity](https://jouledb.org/jsyn))
- **Online Book (`mdBook`):** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb) ([Claude & MCP Apps Guide](https://openie-dev.github.io/jouledb/claude-apps.html))
- **Verified `.jsyn` Sentence Embedder & Benchmark Dataset:** [https://huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)
- **Architecture Whitepaper (`v0.9.0`):** [`docs/WHITEPAPER-JOULEDB-2026-11.md`](./docs/WHITEPAPER-JOULEDB-2026-11.md)
- **`.jsyn` Binary Specification:** [`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](./docs/JSYN-LIVING-WEIGHTS-SPEC.md)
- **Benchmark Scripts & Raw JSON/EDN Outputs:** [`bench/`](./bench/)

## License

Copyright (c) 2026 Open Interface Engineering, Inc. Dual-licensed under either [Apache License, Version 2.0](./LICENSE-APACHE) or [MIT License](./LICENSE-MIT), at your option.
