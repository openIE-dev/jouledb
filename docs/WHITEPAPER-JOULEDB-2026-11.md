# JouleDB: An Energy-Metered Multi-Model Database & Sub-1-Bit Weights Research Substrate

## Architecture & Empirical Status Paper — Version `0.11.0` (Pre-1.0 Research Preview, November 2026)
**Open Interface Engineering, Inc. (`OpenIE`)**  
**Website:** [https://jouledb.org](https://jouledb.org) | **Docs Book:** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb) | **GitHub:** [openIE-dev/jouledb](https://github.com/openIE-dev/jouledb) | **Hugging Face (`.jsyn` Embedder & Raw Benchmarks):** [huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)  
**License:** `Apache-2.0 OR MIT`

> **Revision & Audit Errata Note (`v0.11.0`, 2026-11-26):** Following our `v0.6.0` external-audit remediation (which retracted the premature `v1.0.0` GA tag, removed the unrelated `1.11M`-line `joule-web` crate, deleted synthetic frontier AI exam endpoints, and retracted 81 unverified low-BPW causal LLM exports from Hugging Face) and `v0.7.0`–`v0.10.0` storage/analytical/consensus upgrades, **`v0.11.0`** describes the **`41`-crate (`745,896` LOC, `12,269` tests)** codebase with **`41 / 41` (`100%`, `0` bypasses) `deny(clippy::unwrap_used)` enforcement across the entire workspace**, vectorized out-of-core `.jcol` execution completing **`43 / 43` Official ClickBench queries on the full `99,997,497`-row (`14.78 GB`) dataset in `5.90 GB` post-load RSS (`2.11×` geomean latency & `2.03×` `reported_j` vs DuckDB `2.0.1`, `7 / 43` wins)**, the **Official 22-Query `TPC-H` Suite passing `22 / 22` (`100%`) exact across `SF = 0.1` (`2.22×` geomean, `3` wins), `SF = 1.0` (`8.66M` rows in `2.79 s`, `6.18×` geomean, `6.30 GB` `VmHWM`, `1` win), AND `SF = 10.0` (`86.6M` rows / `60M` `lineitem` out-of-core, `16.03×` geomean)**, 64-shard `JMVR\x01`/`JWB1` `pgwire` OLTP (**`5,797.0 TPS` at `c=4`, `5,038.4 TPS` Simple / `4,990.8 TPS` Pipelined at `c=16`**, `invariant_verified = true`), built-in **Claude/MCP Apps (`SEP-1865`) Studio UI (`ui://jouledb/studio.html`, `50` tools, `.mcpb` bundles)**, and **5-node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex` + `SIGKILL` + `raft.wal`)** verified **`{:valid? true}`** by official Clojure `elle 0.2.8`.

---

## Abstract

Modern data and AI stacks partition state across separate OLTP databases, analytical engines, vector indexes, agent UI sidecars, container sandboxes, and neural inference runtimes. **JouleDB (`v0.11.0`)** explores unifying these workloads inside a single open-source Rust workspace where Copy-on-Write (`CoW`) Serializable Snapshot Isolation (`SSI`) tables, out-of-core `.jcol` vectorized columnar scans & Snowflake multi-table hash joins, PostgreSQL `pgwire` SQL, multi-host WAN TCP Raft `ReadIndex` consensus, `SEP-1865` MCP Apps interactive UI, `10,240`-bit Hyperdimensional Computing (`Amorphic`) holograms, and sub-1-bit factorized sentence-embedding bitplanes (`.jsyn`) share one storage and energy-accounting substrate. Every energy figure is tagged by measurement provenance: **`reported_j`** from bare-metal OS hardware counters (Linux RAPL, macOS IOKit SMC) over multi-millisecond windows, or **`est_j`** from analytical cost models for sub-millisecond operations.

---

## 1. Core Subsystems in `v0.11.0` (`41` Workspace Crates)

1. **Storage, 64-Shard Binary `JMVR`/`JWB1` `SSI` Concurrency & Group-Committed Recovery (`joule-db-core`, `joule-db-local`, `joule-db-amorphic`, `joule-db-server`):**
   - Lock-free positional `pread`/`pwrite`, `IoRingEngine` extent coalescing, Cahill `SSI` (`rw`-antidependency cycle prevention), 64-shard `MvccStore` (`VersionChain::find_visible_at` zero-`HashSet`-clone reads), 64-shard `MvccTableStorage` sessions, `SchemaMetaCache`, binary `JMVR\x01` & `JWB1` WAL serialization, deferred-hologram `batch_ingest_fields`, adaptive `SmallVec`/`BTreeSet` secondary indexes, `commit_with_hook` + `GroupSync` shared `fsync`, `CRC-32` torn-WAL truncation, `ALP + FSST` columnar codecs, `RaBitQ` 1-bit `POPCNT` vector quantization, and double-entry `i128` ledger primitives (`tiger_ledger`).
   - **Compiler Safety Scope:** `#![deny(unsafe_code)]` in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`); `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` enforced in CI across **all `41 / 41` workspace crates (`100%`, `0` `#[allow(clippy::unwrap_used)]` bypasses)** via `cargo clippy --workspace --lib --bins --locked`.
2. **Out-of-Core `.jcol` Vectorized Execution, Columnar Multi-Table `TPC-H` Joins & `pgwire` (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire server (`pgwire` on `port + 2`, Simple & cached-`Describe` Extended Pipeline modes with direct `portal.params` binding, `try_direct_indexed_eq_query` `O(1)`/`O(k)` OLTP fast paths, thread-local SQL AST cache, and §55.2.3 `Sync` recovery), row-budgeted streaming Parquet `COPY` spilling `262,144`-row `.jcol` disk segments, `execute_segmented` (`WordMask` selection vectors, `global_int_sum`, 256-partition `CompactI2Row` / parallel hash-reduce), `columnar_join` streaming `.jcol` segments against post-order Snowflake dimension hash branches (`22 / 22` Official `TPC-H` at `SF=0.1`, `SF=1.0`, and `SF=10.0`), and `DBSP` `ZSet<T>` incremental view maintenance.
3. **Claude Apps & MCP Apps (`SEP-1865`) Host (`mcp_app_ui.rs`, `mcp_transport.rs`):**
   - Exposes `50` MCP tools (`25` core `db.*` incl. `db.studio` + `25` extended) and interactive zero-CDN **JouleDB Studio** (`ui://jouledb/studio.html`, `text/html;profile=mcp-app`) over both `jouledb mcp --data <dir>` (`stdio`) and `POST /mcp` (Streamable HTTP), packaged as one-click Claude Desktop `.mcpb` (`manifest_version: "0.3"`) bundles and an Antigravity plugin.
4. **5-Node Multi-Host WAN & Local TCP Raft Consensus (`raft.rs`, `raft_transport.rs`, `examples/raft_wan_node.rs`):**
   - Quorum write replication via `RaftNode::propose_and_commit` (§5.4.2), quorum linearizable reads via `RaftNode::read_linearizable` (§6.4 `ReadIndex`, zero WAL growth), `CRC-32` durable `raft.wal` persistence, and 2PC participant/coordinator recovery logs.
5. **`.jsyn` (`JSYN-v1`) Sub-1-Bit Weights, `Amorphic` Continuum & Energy Ledger (`joule-db-weights`, `joule-db-amorphic`, `joule-db-energy`, `joule-db-ledger`):**
   - Pure-Rust `LittleBit-2 ⊕ Sakana SVF` bitplanes (`0` PyTorch / ONNX Runtime), `10,240`-bit Kanerva VSA `Amorphic` holograms, $O(1)$ CoW branching (`joule-db-branch`), technical security-control primitives (`trust_compliance.rs`), and Merkle-anchored energy receipts (`reported_j` / `est_j`).

---

## 2. Reproducible External Evaluations (`bench/`)

### 2.1 Full Official ClickBench (`hits.parquet`, `99,997,497` Rows, `105` Columns, `43` Queries) vs. DuckDB `2.0.1`

Evaluated via [`bench/clickbench_official.py`](../bench/clickbench_official.py) on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, 32 threads, `128 GB` RAM, Linux `rapl-package` **`reported_j`**) over real `pgwire` (`psycopg 3.3.9`):

| Official `14.78 GB` `hits.parquet` Scale | Load (`DuckDB` vs `JouleDB`) | JouleDB RSS (`VmHWM` Peak) | Queries Succeeded | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`100,000` rows** | `0.179 s` vs `0.265 s` | `917.8 MB` (`2,651.5 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`34` exact) | **`20`** / `23` | **`1.47×` slower** (`median 1.19×`) | **`1.97×` more J** |
| **`1,000,000` rows** | `0.158 s` vs `0.871 s` | `1,282.9 MB` (`1,922.8 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`11`** / `32` | **`2.34×` slower** (`median 2.95×`) | **`2.50×` more J** |
| **`10,000,000` rows** | `2.422 s` vs `13.340 s` | `2,384.7 MB` (**`3,448.4 MB`**) | **`43 / 43`** (`0` err) | **`43 / 43`** (`31` exact) | **`9`** / `34` | **`2.28×` slower** (`median 2.63×`) | **`2.22×` more J** |
| **`99,997,497` rows (FULL `100M`)** | `13.866 s` vs `97.835 s` | **`5,901.8 MB` (`16,469.7 MB`)** | **`43 / 43`** (`0` err) | **`43 / 43`** (`32` exact) | **`7`** / `36` | **`2.11×` slower** (`median 3.39×`) | **`2.03×` more J** |

### 2.2 Official 22-Query `TPC-H` Relational Suite (`tpch_queries()`, 8 Tables, `SF = 0.1`, `1.0` & `10.0`) vs. DuckDB `2.0.1`

Evaluated via [`bench/tpch_official.py`](../bench/tpch_official.py) ([`tpch_official_sf01.json`](../bench/results/tpch_official_sf01.json), [`tpch_official_sf1.json`](../bench/results/tpch_official_sf1.json), [`tpch_official_sf10.json`](../bench/results/tpch_official_sf10.json)) over real TCP `pgwire` (`psycopg 3.3.9`, Linux `rapl-package` **`reported_j`**):
- **`SF = 0.1` (`866,602` rows across 8 tables):** Parquet load `1.979 s` (`135.3 J`) vs DuckDB `0.328 s` (`13.8 J`); **`22 / 22` (`100%`, `0` errors) verified exact in `318.0 ms` total** vs DuckDB `99.5 ms` (**`2.22×` geomean, `median 1.92×`**, cut **`56.4×`** from `125.50×` in `v0.10.0`), **beating DuckDB `2.0.1` on `3 / 22` queries** (`Q22: 1.31 ms` vs `4.27 ms` [`0.31×`], `Q11: 2.46 ms` vs `4.40 ms` [`0.56×`], `Q20: 3.54 ms` vs `5.66 ms` [`0.63×`]).
- **`SF = 1.0` (`8,661,245` rows across 8 tables, `6,001,215` `lineitem` rows):** Parquet load **`4.884 s` (`256.0 J`, `5,769.3 MB` post-load RSS, `6,303.2 MB` peak `VmHWM`)** vs DuckDB `1.099 s` (`70.9 J`); **`22 / 22` (`100%`, `0` timeouts, `0` errors) verified exact in `2,787.1 ms` (`2.79 s`) total** vs DuckDB `239.4 ms` (**`6.18×` geomean, `median 6.39×`** — eliminating all `9` `v0.10.0` timeouts/RSS aborts: `Q22` **`12.59 ms` vs DuckDB `17.12 ms` [`0.74×` win]**, `Q06: 9.80 ms`, `Q11: 14.99 ms`, `Q17: 15.00 ms`, `Q14: 16.81 ms`, `Q15: 19.13 ms`, `Q19: 21.22 ms`, `Q04: 44.29 ms`, `Q07: 45.82 ms`, `Q08: 52.48 ms`, `Q05: 61.83 ms`).
- **`SF = 10.0` (`86,586,082` rows across 8 tables, `59,986,052` `lineitem` rows out-of-core `.jcol`):** Parquet load **`27.813 s` (`2,026.3 J`, `34.81 GB` post-load RSS, `40.03 GB` peak `VmHWM`)** vs DuckDB `4.964 s` (`530.4 J`); **`22 / 22` (`100%`, `0` errors) verified exact in `34.06 s` total** vs DuckDB `0.980 s` (**`16.03×` geomean, `median 12.44×`**, with **`18 / 22` queries under `1.0 s`**: `Q06: 74.4 ms`, `Q22: 138.0 ms`, `Q19: 149.5 ms`, `Q14: 176.0 ms`, `Q11: 189.0 ms`, `Q15: 205.3 ms`, `Q04: 225.7 ms`, `Q01: 312.1 ms`; candid tail bottlenecks `Q10: 9.11 s` [`126.2×`] & `Q18: 11.33 s` [`94.8×`]).

### 2.3 Head-to-Head `100`-Warehouse TCP `pgwire` OLTP (`100,000` Accounts) vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

Evaluated via [`bench/oltp_pgwire.py`](../bench/oltp_pgwire.py) ([`oltp_pgwire_100wh.json`](../bench/results/oltp_pgwire_100wh.json), quiet host `idle_watts = 10.96 W`) across `1, 4, 8, 16` concurrent `psycopg` TCP clients (`50%` indexed point read, `30%` durable `fsync=on` `BEGIN..UPDATE..INSERT..COMMIT`, `20%` warehouse aggregate; `invariant_verified = true` across all 5 runs):
- **JouleDB `0.11.0` (`pgwire` Simple Query, 4-RTT + `fsync`):** **`4,094.66 TPS`** (`c=1`, `p50 0.134 ms`, `9.38 mJ/txn`) -> **`5,797.01 TPS`** (`c=4`, `p50 0.658 ms`, `9.89 mJ/txn`) -> `5,376.65 TPS` (`c=8`) -> **`5,038.38 TPS`** (`c=16`, `p50 1.370 ms`, `15.93 mJ/txn`, `48,473` writes, `1` SI conflict).
- **JouleDB `0.11.0` (`pgwire` Extended Pipeline, 1-RTT + `fsync`):** **`4,020.27 TPS`** (`c=1`, `p50 0.150 ms`) -> **`5,205.58 TPS`** (`c=4`, `p50 0.780 ms`) -> `5,186.25 TPS` (`c=8`) -> **`4,990.75 TPS`** (`c=16`, `p50 1.384 ms`, `24.55 mJ/txn`, `46,337` writes, `1` SI conflict).
- **PostgreSQL `16.16` (`pgwire` Simple & Extended Pipeline, `fsync=on`, `synchronous_commit=on`, same host/window):** `17,466.14 -> 35,913.17 (c=4) -> 14,676.40 TPS (c=16, 5.18 mJ/txn, 217,684 writes)` in Simple Query mode, and `23,516.32 -> 26,620.99 (c=4) -> 13,029.46 TPS (c=16, 5.66 mJ/txn, 195,399 writes)` in Pipeline mode (`2.91×` / `2.61×` ahead of JouleDB at `c=16`).
- **Embedded `SQLite 3.53.1` (`WAL`, `synchronous=NORMAL`, in-process C API):** `90,161.56 -> 127,046.29 TPS` (`0.44 mJ/txn`, `1,035,821` writes, no TCP wire hops or per-commit `fsync`).

### 2.4 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

Evaluated via [`bench/raft_multihost_wan.py`](../bench/raft_multihost_wan.py) ([`raft_wan_multihost_summary.json`](../bench/results/raft_wan_multihost_summary.json)) and [`bench/elle_check.clj`](../bench/elle_check.clj):
- **Cross-Machine WAN Faults:** 5 separate OS `raft_wan_node` processes across **2 distinct physical Linux hosts** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM, **`27.58 ms` median Tailscale WAN TCP RTT**); cross-WAN `2/5` minority split isolating `n1` -> stale write & `ReadIndex` rejection -> `3/5` majority elects `n3` (term `2`, `24` ops) -> real OS `SIGKILL` (`kill -9`) on `n5` (`16` ops during crash) -> `raft.wal` restart & 5-node convergence (`commit=68, applied=68, x=68, y=39`) -> **`{:valid? true}`** (`rw-register`, `:strict-serializable`, `270` WAN events).

### 2.5 `.jsyn` Compression vs. Fidelity Boundaries

- **Verified High-Fidelity Encoder (`0.4378 BPW`):** [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`Apache-2.0`, `1.95 MB`, `22.23×` smaller than `FP16`) achieves **`0.9441` mean sentence-embedding cosine similarity** against upstream `FP32`.
- **Degraded Ultra-Low-BPW Causal LLMs (`0.02–0.17 BPW`, Retracted):** Out-of-sample rank sweeps (`r = 8..128`) on `Gemma-4-E2B` (`q_proj`) and `Inkling-2-Mini` (`128`-Exp MoE) yield **`0.113–0.455` output cosine** (`KL = 2.59–3.87`, `65.6%` Top-5 agreement at `r = 128`). All 81 unverified causal LLM checkpoints remain deleted from Hugging Face.

---

## 3. Explicit Engineering & Governance Boundaries

1. **Pre-1.0 Research Preview (`v0.11.0`):** `100%` (`41 / 41`) `deny(clippy::unwrap_used)` coverage, out-of-core `.jcol` segments (`2.11×` geomean vs DuckDB `2.0.1` on `100M` ClickBench), `22 / 22` Official `TPC-H` up to `SF=10.0` (`86.6M` rows), and `5,797 TPS` 64-shard `pgwire` OLTP substantially advance verified engine maturity, while mature specialized engines (`DuckDB 2.0.1` `2.11×` faster on `100M` ClickBench & `6.18×`–`16.03×` faster on `SF=1..10` `TPC-H`; `PostgreSQL 16.16` `2.6×–6.2×` faster on `pgwire` OLTP; `SQLite` embedded WAL) remain ahead on single-model OLAP/OLTP throughput.
2. **No Third-Party Compliance Certifications:** `trust_compliance.rs` implements technical controls (`external_audit_performed = false`). **No SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been performed.**
3. **SDK Artifacts & Portable Binaries:** Prebuilt `v0.11.0` server archives, Claude `.mcpb` bundles, Python `cp39-abi3` `.whl` wheels, and TypeScript `openie-jouledb-0.11.0.tgz` ship on GitHub Releases (`GLIBC <= 2.35` on Linux); client SDKs are not yet published on PyPI, npm, or NuGet pending registry token/OIDC registration.
