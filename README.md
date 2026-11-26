# JouleDB (`v0.11.0` — Pre-1.0 Research Preview)

**An experimental energy-metered multi-model database in Rust (`Apache-2.0 OR MIT`) exploring Copy-on-Write `SSI` storage, out-of-core `.jcol` vectorized columnar analytics (`43 / 43` Official ClickBench at `100M` rows + `22 / 22` Official `TPC-H` at `SF=0.1`, `SF=1.0`, & `SF=10.0` [`86.6M` rows]), 64-shard `pgwire` SQL OLTP, multi-host WAN TCP Raft `ReadIndex` consensus, Claude/MCP Apps Studio UI (`SEP-1865`), `10,240`-bit `Amorphic` VSA storage, and `.jsyn` sub-1-bit sentence embeddings.**

[![Release](https://github.com/openIE-dev/jouledb/actions/workflows/release.yml/badge.svg)](https://github.com/openIE-dev/jouledb/releases)
[![Quality](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml/badge.svg)](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml)
[![Docs](https://github.com/openIE-dev/jouledb/actions/workflows/docs.yml/badge.svg)](https://openie-dev.github.io/jouledb)
[![crates.io](https://img.shields.io/crates/v/jouledb.svg)](https://crates.io/crates/jouledb)
[![Hugging Face Embedder](https://img.shields.io/badge/Hugging%20Face-MiniLM%20.jsyn%20(cos%200.944)-0D9488)](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)
[![Website](https://img.shields.io/badge/Web-jouledb.org-2563EB)](https://jouledb.org)

> **Pre-1.0 Status & `v0.11.0` Release Summary (`2026-11-26`):** Building on our `v0.6.0` external-audit remediation (`v1.0.0` GA tag retracted/yanked, unrelated `1.11M`-LOC `joule-web` crate and synthetic `frontier_exam_routes.rs` deleted, 81 unverified low-BPW causal LLM exports pruned from Hugging Face) and `v0.7.0`–`v0.10.0` engine upgrades, **`v0.11.0`** reaches **`41 / 41` (`100%`) workspace `deny(clippy::unwrap_used)` enforcement**, completes **`22 / 22` (`100%`) Official `TPC-H` queries exact across `SF = 0.1`, `SF = 1.0` (`8.66M` rows in `2.79 s`), AND `SF = 10.0` (`86.6M` rows / `60M` `lineitem` out-of-core!)**, and lifts **`100`-warehouse `pgwire` OLTP to `5,797.0 TPS` (`c=4`) / `5,038.4 TPS` (`c=16`, `2.61×`–`2.91×` vs `PostgreSQL 16.16`)**:
> - **`41 / 41` Workspace Crates (`100%`) Enforcing `deny(clippy::unwrap_used)` (`0` Bypasses):** Eliminated all remaining non-test `.unwrap()` calls across the final 12 crates (`joule-db-amorphic`, `joule-db-features`, `joule-db-hdc`, `joule-db-browser`, `joule-db-domains`, `joule-db-gpu`, `joule-runtime`, `sigql`, `inv-auth`, `inv-crypto`, `inv-energy`, `inv-telemetry`), enforcing a workspace-wide `cargo clippy --workspace --lib --bins --locked` gate in `.github/workflows/engine.yml`.
> - **Out-of-Core Columnar Multi-Table Join & Subquery Engine (`columnar_join.rs` — `22 / 22` Official `TPC-H` up to `SF = 10.0`, `86.6M` Rows):** Streams `262,144`-row `.jcol` fact segments directly against compact post-order Snowflake dimension hash tables (`DenseKeyBits`, `NeqSummary`, correlated scalar domain pushdown), cutting `SF = 0.1` geomean from **`125.50× -> 2.22×`** (`3 / 22` faster than DuckDB `2.0.1`), upgrading `SF = 1.0` (`8.66M` rows) from `13 / 22` (`9` timeouts/OOM) to **`22 / 22` exact in `2.79 s` total (`6.18×` geomean, `6.30 GB` peak `VmHWM`, `Q22` `1.36×` faster than DuckDB)**, and passing **`22 / 22` exact at `SF = 10.0` (`86,586,082` rows, `16.03×` geomean)**.
> - **64-Shard `MvccStore` + Deferred-Hologram Binary `JWB1` `pgwire` OLTP (`oltp_pgwire_100wh.json`):** 64-shard `MvccStore` (`find_visible_at` zero-clone reads), 64-shard `MvccTableStorage` sessions, `SchemaMetaCache`, `O(1)`/`O(k)` direct indexed `SELECT`/`COUNT`/`SUM` fast paths, `O(1)` `deferred_set`, adaptive `BTreeSet` `ValueIds`, and zero-JSON `batch_ingest_fields` reach **`4,094.7 TPS` (`c=1`, `p50 0.134 ms`) -> `5,797.0 TPS` (`c=4`) -> `5,038.4 TPS` (`c=16` Simple) / `4,990.8 TPS` (`c=16` Pipelined)** with `invariant_verified = true` across all 5 runs.
> - **Full `99,997,497`-Row (`14.78 GB`) Official ClickBench (`2.11×` Geomean vs DuckDB `2.0.1`), 5-Node Multi-Host WAN Raft (`elle` `{:valid? true}`), & Claude/MCP Apps (`.mcpb`):** Built-in **JouleDB Studio (`ui://jouledb/studio.html`, `50` tools)**, `.mcpb` bundles, `cp39-abi3` `.whl` wheels, and `openie-jouledb-0.11.0.tgz` on GitHub Release `v0.11.0`. See [`CHANGELOG.md`](./CHANGELOG.md).

---

## What Is Real & Verified Today (`41` Workspace Crates · `745.9k` LOC · `12,269` Tests · `41 / 41` `deny(unwrap_used)`)

1. **Core Storage, 64-Shard Binary `JMVR` MVCC `SSI` & Recovery (`joule-db-core`, `joule-db-local`, `joule-db-amorphic`):**
   - Copy-on-Write B-tree pages, lock-free positional `pread`/`pwrite`, Cahill Serializable Snapshot Isolation (`SSI` `rw`-antidependency detection), 64-shard `MvccStore` (`find_visible_at` zero-`HashSet`-clone snapshot reads) + 64-shard `MvccTableStorage` with binary `JMVR\x01` & `JWB1` WAL serialization, deferred-hologram `batch_ingest_fields`, adaptive `SmallVec`/`BTreeSet` secondary indexes, group-committed WAL durability, `CRC-32` torn-WAL truncation, `ALP + FSST` columnar codecs, and `RaBitQ` 1-bit `POPCNT` vector quantization.
2. **Out-of-Core `.jcol` Columnar Analytics, Vectorized Multi-Table `TPC-H` Joins & `pgwire` (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire listener (`psql`, `psycopg3` Simple & cached-`Describe` Extended Pipeline with `SchemaMetaCache` and thread-local AST cache), row-budgeted streaming Parquet `COPY` spilling `262,144`-row `.jcol` disk segments, vectorized `execute_segmented` (`43 / 43` Official ClickBench at `100M` rows) + `columnar_join` (`22 / 22` Official `TPC-H` at `SF=0.1`, `SF=1.0`, and `SF=10.0`), and `DBSP` `ZSet<T>` incremental views.
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
| **`100,000` rows** ([JSON](./bench/results/clickbench_official_100k.json)) | `0.179 s` vs `0.265 s` | `918 MB` (`2,652 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`34` exact) | **`20`** / `23` | **`1.47×` slower** (`median 1.19×`) | **`1.97×` more J** |
| **`1,000,000` rows** ([JSON](./bench/results/clickbench_official_1m.json)) | `0.158 s` vs `0.871 s` | `1,283 MB` (`1,923 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`11`** / `32` | **`2.34×` slower** (`median 2.95×`) | **`2.50×` more J** |
| **`10,000,000` rows** ([JSON](./bench/results/clickbench_official_10m.json)) | `2.422 s` vs `13.340 s` | `2,385 MB` (**`3,448 MB`**) | **`43 / 43`** (`0` err) | **`43 / 43`** (`31` exact) | **`9`** / `34` | **`2.28×` slower** (`median 2.63×`) | **`2.22×` more J** |
| **`99,997,497` rows — FULL `14.78 GB`** ([JSON](./bench/results/clickbench_official_100m.json)) | `13.866 s` vs `97.835 s` | **`5,902 MB` (`16,470 MB`)** | **`43 / 43`** (`0` err) | **`43 / 43`** (`32` exact) | **`7`** / `36` | **`2.11×` slower** (`median 3.39×`) | **`2.03×` more J** |

- **Candid Full-`100M` Summary:** Vectorized `.jcol` execution (`5.90 GB` post-load RSS; `10M` peak `VmHWM` `3.45 GB`) narrows the full `100M`-row geomean latency gap vs DuckDB `2.0.1` from `4.85×` (`v0.8.0`) -> `2.82×` (`v0.9.0`) -> **`2.11×`** (`2.03×` `reported_j`), beating DuckDB `2.0.1` on **`7 / 43` queries at `100M` rows** (`Q0` `1.91 ms`, `Q1` `20.23 ms`, `Q2` `2.41 ms` [`25.5×`], `Q3` `2.32 ms` [`22.4×`], `Q6` `1.58 ms` [`14.0×`], `Q29` `2.40 ms` [`23.4×`], `Q42` `22.78 ms`). DuckDB `2.0.1` wins **`36 / 43` queries (`2.11×` geomean, `7.06×` faster Parquet load)**.

### 2. Official 22-Query `TPC-H` Relational Suite (`tpch_queries()`, 8 Tables, `SF = 0.1`, `1.0` & `10.0`) vs. DuckDB `2.0.1`

Executed via [`bench/tpch_official.py`](./bench/tpch_official.py) over real TCP `pgwire` (`psycopg 3.3.9`, Linux `rapl-package` **`reported_j`**) across all **22 official `TPC-H` SQL queries (`Q01..Q22`)**:

| Official `TPC-H` Scale | 8-Table Parquet Load (`DuckDB` vs `JouleDB`) | JouleDB Peak `VmHWM` | Exact Verified (`Faster`) | Total 22-Query Time (`DuckDB` vs `JouleDB`) | Geomean Latency (`JouleDB / DuckDB`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`SF = 0.1` (`866,602` rows)** ([JSON](./bench/results/tpch_official_sf01.json)) | `0.328 s` (`13.8 J`) vs `1.979 s` (`135.3 J`) | `2,205 MB` | **`22 / 22`** (`3` faster: `Q11,Q20,Q22`) | `99.5 ms` vs **`318.0 ms`** (`155.9×` vs `v0.10.0`) | **`2.22×` slower** (`median 1.92×`, down from `125.5×`) |
| **`SF = 1.0` (`8,661,245` rows)** ([JSON](./bench/results/tpch_official_sf1.json)) | `1.099 s` (`70.9 J`) vs `4.884 s` (`256.0 J`) | **`6,303 MB`** | **`22 / 22`** (`1` faster: `Q22` `0.74×`) | `239.4 ms` vs **`2,787.1 ms`** (`0` timeouts!) | **`6.18×` slower** (`median 6.39×`, vs `13/22` in `v0.10.0`) |
| **`SF = 10.0` (`86,586,082` rows, `60M` `lineitem`)** ([JSON](./bench/results/tpch_official_sf10.json)) | `4.964 s` (`530.4 J`) vs `27.813 s` (`2,026.3 J`) | `40,031 MB` | **`22 / 22`** (`0` faster, `0` err) | `979.8 ms` vs **`34,060.0 ms`** (`18/22 < 1 s`) | **`16.03×` slower** (`median 12.44×`) |

- **Candid `TPC-H` Summary:** `columnar_join.rs` streams `.jcol` segments directly against compact Snowflake dimension hash tables, eliminating all `9` `SF=1.0` timeouts/OOMs from `v0.10.0` (`Q05` `61.8 ms`, `Q07` `45.8 ms`, `Q08` `52.5 ms`, `Q14` `16.8 ms`, `Q17` `15.0 ms`, `Q22` **`12.59 ms` vs DuckDB `17.12 ms`**) and completing **`22 / 22` exact on `86,586,082` rows (`SF=10.0`)**. DuckDB `2.0.1` remains **`6.18×` faster at `SF=1.0`** and **`16.03×` faster at `SF=10.0`** (widest gaps on `SF=10.0`: `Q10` `9.11 s` [`126.2×`] and `Q18` `11.33 s` [`94.8×`]).

### 3. Head-to-Head `100`-Warehouse TCP `pgwire` OLTP vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

Executed via [`bench/oltp_pgwire.py`](./bench/oltp_pgwire.py) ([`oltp_pgwire_100wh.json`](./bench/results/oltp_pgwire_100wh.json), `rapl-package` **`reported_j`**, quiet host `idle_watts = 10.96 W`): `100,000` accounts (`100` warehouses), `50%` indexed point read, `30%` durable `BEGIN; UPDATE accounts ...; INSERT INTO history ...; COMMIT` (`fsync=on`, `synchronous_commit=on`), `20%` warehouse `COUNT`/`SUM` across `1, 4, 8, 16` concurrent TCP clients:

| Engine & Wire Protocol (`invariant_verified = true` on all 5) | `c = 1` TPS (`p50`) | `c = 4` TPS (`p50`) | `c = 8` TPS (`p50`) | `c = 16` TPS (`p50`, `mJ/txn`) | Committed Writes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **JouleDB `0.11.0` (`pgwire` Simple Query, 4-RTT + `fsync`)** | `4,094.7` (`0.134 ms`) | **`5,797.0`** (`0.658 ms`) | `5,376.7` (`0.997 ms`) | **`5,038.4`** (`1.370 ms`, `15.93 mJ`) | `48,473` (`1` SI conflict) |
| **JouleDB `0.11.0` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `4,020.3` (`0.150 ms`) | **`5,205.6`** (`0.780 ms`) | `5,186.3` (`1.056 ms`) | **`4,990.8`** (`1.384 ms`, `24.55 mJ`) | `46,337` (`1` SI conflict) |
| **PostgreSQL `16.16` (`pgwire` Simple Query, 4-RTT + `fsync`)** | `17,466.1` (`0.064 ms`) | `35,913.2` (`0.097 ms`) | `22,680.3` (`0.247 ms`) | **`14,676.4`** (`0.723 ms`, `5.18 mJ`) | `217,684` (`0` conflicts) |
| **PostgreSQL `16.16` (`pgwire` Extended Pipeline, 1-RTT + `fsync`)** | `23,516.3` (`0.041 ms`) | `26,621.0` (`0.132 ms`) | `18,251.5` (`0.347 ms`) | **`13,029.5`** (`0.975 ms`, `5.66 mJ`) | `195,399` (`0` conflicts) |
| **SQLite `3.53.1` (Embedded in-proc C API, `WAL`, `NORMAL`)** | `90,161.6` (`0.005 ms`) | `104,040.7` (`0.005 ms`) | `109,016.6` (`0.006 ms`) | **`127,046.3`** (`0.006 ms`, `0.44 mJ`) | `1,035,821` (no TCP / `fsync`) |

- **Candid OLTP Summary:** 64-shard `MvccStore`/`MvccTableStorage`, `SchemaMetaCache`, direct indexed `O(1)`/`O(k)` reads, adaptive `BTreeSet` `ValueIds`, and deferred-hologram `batch_ingest_fields` lift JouleDB `pgwire` throughput to **`5,797.0 TPS` (`c=4`) / `5,038.4 TPS` (`c=16`, `0.134 ms` `c=1` `p50`)**, narrowing the `c=16` gap vs `PostgreSQL 16.16` to **`2.91×` Simple (`2.61×` Pipelined)**. `PostgreSQL 16.16` still wins `2.6×–6.2×` across concurrency levels via shared-buffer page locks vs JouleDB's store-wide write lock during commit apply.

### 4. 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

Executed via [`bench/raft_multihost_wan.py`](./bench/raft_multihost_wan.py), [`examples/raft_wan_node.rs`](./crates/joule-db-server/examples/raft_wan_node.rs), and [`bench/elle_check.clj`](./bench/elle_check.clj):
- **Cross-Machine WAN Topology & Nemeses:** 5 real OS `raft_wan_node` processes with durable `raft.wal` across **2 distinct physical Linux hosts** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM, **`27.58 ms` median Tailscale WAN TCP RTT**), cross-WAN `2/5` minority split isolating `n1`, `3/5` majority election (`n3`), real OS **`SIGKILL` (`kill -9`)** on `n5`, and `raft.wal` convergence (`commit=68, applied=68, x=68, y=39`) -> **[`{:valid? true}`](./bench/results/raft_wan_multihost_elle_verdict.txt)** (`270` WAN events, `:strict-serializable`).

### 5. `.jsyn` (`JSYN-v1`) Fidelity vs. Compression Curve

| Model / Layer | `BPW` | Shrink vs `FP16` | Measured Fidelity vs Upstream | Guidance |
| :--- | :---: | :---: | :---: | :--- |
| **[`all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit)** (`6L` Encoder) | `0.438` | `22.2×` (`1.95 MB`) | **`0.9441` embedding cosine** vs `FP32` | **Recommended** for semantic search & `TriModal RRF` |
| `Gemma-4-E2B` `q_proj` (`r = 8..128`) | `0.028–0.166` | `564×–96.6×` | `0.271–0.455` output cos (`KL 3.87–2.72`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |
| `Inkling-2-Mini` `w13` (`128`-Exp MoE, `r = 8..128`) | `0.023–0.134` | `701×–120×` | `0.113–0.350` output cos (`KL 2.91–2.59`) | **Experimental only** — not a drop-in causal LLM (retracted from HF) |

---

## Quickstart (`v0.11.0`)

```bash
# 1. Install prebuilt v0.11.0 binary (macOS ARM64/x86_64; Linux x86_64 GLIBC <= 2.35, Ubuntu 22.04/24.04 & Debian 12 ready)
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

- **Pre-1.0 research preview (`v0.11.0`), not a battle-tested production RDBMS:** While JouleDB `0.11.0` executes `43 / 43` Official ClickBench on `99,997,497` rows (`2.11×` geomean vs DuckDB `2.0.1`), `22 / 22` Official `TPC-H` up to `SF=10.0` (`86.6M` rows), `5,797 TPS` `pgwire` OLTP (`invariant_verified = true`), and 5-node cross-machine WAN Raft (`elle` `{:valid? true}`), mature specialized engines (`DuckDB 2.0.1` `2.11×` faster on `100M` ClickBench & `6.18×`–`16.03×` faster on `SF=1..10` `TPC-H`; `PostgreSQL 16.16` `2.6×–6.2×` faster on concurrent `pgwire` OLTP; `SQLite` embedded WAL) remain faster for conventional single-model production workloads.
- **No external compliance certifications:** `jouledb trust` exercises technical security primitives in code (`external_audit_performed = false`). **No third-party SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been conducted.**
- **Exact compiler safety lint scope:** `#![deny(unsafe_code)]` is set in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`). `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` is enforced across **`41 / 41` workspace crates (`100%`, `0` `#[allow(clippy::unwrap_used)]` bypasses)** via a blocking CI gate.
- **SDK registry status:** Prebuilt `cp39-abi3` Python `.whl` wheels, TypeScript `openie-jouledb-0.11.0.tgz`, and Claude `.mcpb` bundles are published on [GitHub Releases (`v0.11.0`)](https://github.com/openIE-dev/jouledb/releases/tag/v0.11.0) (with automated OIDC/token publish steps wired in `.github/workflows/release.yml`); packages are **not yet published** on PyPI, npm, NuGet, or Maven Central pending registry token/OIDC registration.

---

## Documentation & Links

- **Website:** [https://jouledb.org](https://jouledb.org) ([Claude & MCP Apps](https://jouledb.org/claude) · [Benchmarks](https://jouledb.org/benchmarks) · [Honest Boundaries](https://jouledb.org/honest) · [Security Controls & Bridges](https://jouledb.org/trust) · [`.jsyn` Fidelity](https://jouledb.org/jsyn))
- **Online Book (`mdBook`):** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb) ([Claude & MCP Apps Guide](https://openie-dev.github.io/jouledb/claude-apps.html))
- **Verified `.jsyn` Sentence Embedder & Benchmark Dataset:** [https://huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)
- **Architecture Whitepaper (`v0.11.0`):** [`docs/WHITEPAPER-JOULEDB-2026-11.md`](./docs/WHITEPAPER-JOULEDB-2026-11.md)
- **`.jsyn` Binary Specification:** [`docs/JSYN-LIVING-WEIGHTS-SPEC.md`](./docs/JSYN-LIVING-WEIGHTS-SPEC.md)
- **Benchmark Scripts & Raw JSON/EDN Outputs:** [`bench/`](./bench/)

## License

Copyright (c) 2026 Open Interface Engineering, Inc. Dual-licensed under either [Apache License, Version 2.0](./LICENSE-APACHE) or [MIT License](./LICENSE-MIT), at your option.
