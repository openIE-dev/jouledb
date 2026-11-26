# JouleDB: An Energy-Metered Multi-Model Database & Sub-1-Bit Weights Research Substrate

## Architecture & Empirical Status Paper — Version `0.10.0` (Pre-1.0 Research Preview, November 2026)
**Open Interface Engineering, Inc. (`OpenIE`)**  
**Website:** [https://jouledb.org](https://jouledb.org) | **Docs Book:** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb) | **GitHub:** [openIE-dev/jouledb](https://github.com/openIE-dev/jouledb) | **Hugging Face (`.jsyn` Embedder & Raw Benchmarks):** [huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)  
**License:** `Apache-2.0 OR MIT`

> **Revision & Audit Errata Note (`v0.10.0`, 2026-11-26):** Following our `v0.6.0` external-audit remediation (which retracted the premature `v1.0.0` GA tag, removed the unrelated `1.11M`-line `joule-web` crate, deleted synthetic frontier AI exam endpoints, and retracted 81 unverified low-BPW causal LLM exports from Hugging Face) and `v0.7.0`–`v0.9.0` storage/analytical/consensus upgrades, **`v0.10.0`** describes the **`41`-crate (`737,265` LOC, `12,269` tests)** codebase with **`29 / 41` (`70.7%`) `deny(clippy::unwrap_used)` crates (including all of `joule-db-server` & `jouledb-cli`)**, vectorized out-of-core `.jcol` execution completing **`43 / 43` Official ClickBench queries on the full `99,997,497`-row (`14.78 GB`) dataset in `5.90 GB` post-load RSS (`2.11×` geomean latency & `2.03×` `reported_j` vs DuckDB `2.0.1`, `7 / 43` wins; `10M` `VmHWM` cut `2.63×` to `3.45 GB`)**, the **Official 22-Query `TPC-H` Suite (`22 / 22` exact at `SF = 0.1`, `13 / 22` exact at `SF = 1.0`)**, binary `JMVR\x01` MVCC + cached-`Describe` `pgwire` OLTP (`1,375.7 TPS` Extended Pipeline matching `1,351.6 TPS` Simple at `c=16`, `invariant_verified = true`), built-in **Claude/MCP Apps (`SEP-1865`) Studio UI (`ui://jouledb/studio.html`, `50` tools, `.mcpb` bundles)**, and **5-node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex` + `SIGKILL` + `raft.wal`)** verified **`{:valid? true}`** by official Clojure `elle 0.2.8`.

---

## Abstract

Modern data and AI stacks partition state across separate OLTP databases, analytical engines, vector indexes, agent UI sidecars, container sandboxes, and neural inference runtimes. **JouleDB (`v0.10.0`)** explores unifying these workloads inside a single open-source Rust workspace where Copy-on-Write (`CoW`) Serializable Snapshot Isolation (`SSI`) tables, out-of-core `.jcol` vectorized columnar scans, multi-table relational joins, PostgreSQL `pgwire` SQL, multi-host WAN TCP Raft `ReadIndex` consensus, `SEP-1865` MCP Apps interactive UI, `10,240`-bit Hyperdimensional Computing (`Amorphic`) holograms, and sub-1-bit factorized sentence-embedding bitplanes (`.jsyn`) share one storage and energy-accounting substrate. Every energy figure is tagged by measurement provenance: **`reported_j`** from bare-metal OS hardware counters (Linux RAPL, macOS IOKit SMC) over multi-millisecond windows, or **`est_j`** from analytical cost models for sub-millisecond operations.

---

## 1. Core Subsystems in `v0.10.0` (`41` Workspace Crates)

1. **Storage, Binary `JMVR` `SSI` Concurrency & Group-Committed Recovery (`joule-db-core`, `joule-db-local`, `joule-db-server`):**
   - Lock-free positional `pread`/`pwrite`, `IoRingEngine` extent coalescing, Cahill `SSI` (`rw`-antidependency cycle prevention), index-assisted `MvccTableStorage` with binary `JMVR\x01` `MvccRecord` serialization, `O(1)` PK/UNIQUE index checks, `commit_with_hook` + `GroupSync` shared `fsync`, `CRC-32` torn-WAL truncation, LSN-ordered replay, `ALP + FSST` columnar codecs, `RaBitQ` 1-bit `POPCNT` vector quantization, and double-entry `i128` ledger primitives (`tiger_ledger`).
   - **Compiler Safety Scope:** `#![deny(unsafe_code)]` in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`); `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` enforced in CI across **`29 / 41` workspace crates (`70.7%`)** — including all of `joule-db-server` (`lib` + `bin`), `jouledb-cli`, `joule-db-core`, `joule-db-query`, `joule-db-local`, `joule-db-amorphic`, `joule-db-hdc`, `joule-db-viz`, `joule-db-branch`, `joule-db-ledger`, `joule-db-energy`, `joule-db-weights`, and 17 domain/transport crates — with `warn` across the remaining `12 / 41` crates.
2. **Out-of-Core `.jcol` Vectorized Execution v2, Multi-Table `TPC-H` Joins & `pgwire` (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire server (`pgwire` on `port + 2`, Simple & cached-`Describe` Extended Pipeline modes with direct `portal.params` binding, thread-local 256-entry LRU SQL AST cache, and §55.2.3 `Sync` recovery), row-budgeted streaming Parquet `COPY` spilling `262,144`-row `.jcol` disk segments (`I8/I16/I32/I64` narrowing, `global_int_sum` + ZoneMap headers, bitpacked null masks, compact `u8/u16/u32` dictionaries, CRC-32 slice + `JCL1` footer checksums), `execute_segmented` (`WordMask` bitpacked selection vectors, filtered `MIN`/`MAX`, `90`-sum SIMD slice batching, dictionary `LIKE` bitsets, 256-partition `CompactI2Row` / parallel hash-reduce), decorrelated `EXISTS` / `NOT EXISTS` / `IN` hash semi/anti-joins (`22 / 22` Official `TPC-H` at `SF=0.1`), and `DBSP` `ZSet<T>` incremental view maintenance.
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

### 2.2 Official 22-Query `TPC-H` Relational Suite (`tpch_queries()`, 8 Tables) vs. DuckDB `2.0.1`

Evaluated via [`bench/tpch_official.py`](../bench/tpch_official.py) ([`tpch_official_sf01.json`](../bench/results/tpch_official_sf01.json), [`tpch_official_sf1.json`](../bench/results/tpch_official_sf1.json)) over real TCP `pgwire` (`psycopg 3.3.9`, Linux `rapl-package` **`reported_j`**):
- **`SF = 0.1` (`866,602` rows across 8 tables):** Parquet load `4.138 s` (`533.4 J`) vs DuckDB `0.423 s` (`49.2 J`); **`22 / 22` (`100%`, `0` errors) verified exact** (`Q06: 5.00 ms`, `Q15: 5.65 ms`, `Q01: 62.57 ms`, `Q10: 144.56 ms`, `Q14: 219.56 ms`, `Q13: 245.66 ms`, `Q22: 258.24 ms`; `125.50×` geomean vs DuckDB `2.0.1`; `3 / 22` faster than DuckDB at `SF = 0.01`).
- **`SF = 1.0` (`8,661,245` rows across 8 tables, `6,001,215` `lineitem` rows):** Parquet load **`9.816 s` (`1,202.3 J`)** vs DuckDB `2.076 s` (`277.1 J`); **`13 / 22` verified exact** (`Q15: 70.83 ms`, `Q01: 84.18 ms`, `Q06: 336.84 ms`, `Q14: 2,405.5 ms`, `Q20: 2,797.2 ms`, `Q03: 2,871.6 ms`, `Q19: 3,393.5 ms`, `Q10: 3,496.5 ms`, `Q22: 3,995.5 ms`, `Q16: 4,205.7 ms`, `Q11: 4,683.5 ms`, `Q13: 7,538.6 ms`, `Q02: 14,794.2 ms`; `112.75×` geomean across `13` completed queries), while `8 / 22` (`Q04, Q07, Q08, Q09, Q12, Q17, Q18, Q21`) exceeded the `25 s` wall cap and `1 / 22` (`Q05`) tripped the `36 GB` RSS guard due to row-oriented `Vec<serde_json::Value>` join materialization.

### 2.3 Head-to-Head `100`-Warehouse TCP `pgwire` OLTP (`100,000` Accounts) vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

Evaluated via [`bench/oltp_pgwire.py`](../bench/oltp_pgwire.py) ([`oltp_pgwire_100wh.json`](../bench/results/oltp_pgwire_100wh.json), measured back-to-back under shared host CI load `idle_watts = 145.12 W`) across `1, 4, 8, 16` concurrent `psycopg` TCP clients (`50%` indexed point read, `30%` durable `fsync=on` `BEGIN..UPDATE..INSERT..COMMIT`, `20%` warehouse aggregate; `invariant_verified = true` across all 5 runs):
- **JouleDB `0.10.0` (`pgwire` Simple Query, 4-RTT + `fsync`):** `721.59 TPS` (`c=1`, `p50 1.208 ms`) -> **`1,439.95 TPS`** (`c=4`, `p50 1.682 ms`) -> `1,358.36 TPS` (`c=8`) -> `1,351.57 TPS` (`c=16`, `101.6 mJ/txn`, `11,537` writes, `0` errors; uncontended host in `v0.9.0`: `3,351.04 TPS`).
- **JouleDB `0.10.0` (`pgwire` Extended Pipeline, 1-RTT + `fsync`):** `381.88 TPS` (`c=1`) -> `1,105.94 TPS` (`c=4`) -> `1,370.54 TPS` (`c=8`) -> **`1,375.67 TPS`** (`c=16`, `99.5 mJ/txn`, `10,044` writes, `1` SI conflict — **eliminating the `1.97×` Extended Pipeline penalty vs Simple Query** via cached `Describe` rows, direct `portal.params`, AST cache, and binary `JMVR\x01` `MvccRecord` codec).
- **PostgreSQL `16.16` (`pgwire` Simple & Extended Pipeline, `fsync=on`, `synchronous_commit=on`, same host/window):** `836.71 -> 5,744.85 TPS` (`22.8 mJ/txn` at `c=16`, `30,605` writes) in Simple Query mode, and `896.62 -> 4,720.52 TPS` (`27.4 mJ/txn`, `26,462` writes) in Pipeline mode.
- **Embedded `SQLite 3.53.1` (`WAL`, `synchronous=NORMAL`, in-process C API):** `19,826.03 -> 36,085.06 TPS` (`3.49 mJ/txn`, `303,731` writes, no TCP wire hops or per-commit `fsync`).

### 2.4 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

Evaluated via [`bench/raft_multihost_wan.py`](../bench/raft_multihost_wan.py) ([`raft_wan_multihost_summary.json`](../bench/results/raft_wan_multihost_summary.json)) and [`bench/elle_check.clj`](../bench/elle_check.clj):
- **Cross-Machine WAN Faults:** 5 separate OS `raft_wan_node` processes across **2 distinct physical Linux hosts** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM, **`27.58 ms` median Tailscale WAN TCP RTT**); cross-WAN `2/5` minority split isolating `n1` -> stale write & `ReadIndex` rejection -> `3/5` majority elects `n3` (term `2`, `24` ops) -> real OS `SIGKILL` (`kill -9`) on `n5` (`16` ops during crash) -> `raft.wal` restart & 5-node convergence (`commit=68, applied=68, x=68, y=39`) -> **`{:valid? true}`** (`rw-register`, `:strict-serializable`, `270` WAN events).

### 2.5 `.jsyn` Compression vs. Fidelity Boundaries

- **Verified High-Fidelity Encoder (`0.4378 BPW`):** [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`Apache-2.0`, `1.95 MB`, `22.23×` smaller than `FP16`) achieves **`0.9441` mean sentence-embedding cosine similarity** against upstream `FP32`.
- **Degraded Ultra-Low-BPW Causal LLMs (`0.02–0.17 BPW`, Retracted):** Out-of-sample rank sweeps (`r = 8..128`) on `Gemma-4-E2B` (`q_proj`) and `Inkling-2-Mini` (`128`-Exp MoE) yield **`0.113–0.455` output cosine** (`KL = 2.59–3.87`, `65.6%` Top-5 agreement at `r = 128`). All 81 unverified causal LLM checkpoints remain deleted from Hugging Face.

---

## 3. Explicit Engineering & Governance Boundaries

1. **Pre-1.0 Research Preview (`v0.10.0`):** Out-of-core `.jcol` segments (`5.90 GB` load RSS, `2.11×` geomean vs DuckDB `2.0.1` on `100M` rows), `22 / 22` Official `TPC-H` at `SF=0.1` (`13 / 22` at `SF=1.0`), and binary `JMVR` + cached-`Describe` `pgwire` pipelining substantially expand verified capabilities, while mature specialized engines (`DuckDB 2.0.1` on vectorized multi-table `TPC-H` joins & `2.11×` faster `100M` ClickBench; `PostgreSQL 16.16` at `c=16` `pgwire` OLTP; `SQLite` embedded WAL) remain ahead on single-model OLAP/OLTP throughput.
2. **No Third-Party Compliance Certifications:** `trust_compliance.rs` implements technical controls (`external_audit_performed = false`). **No SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been performed.**
3. **SDK Artifacts & Portable Binaries:** Prebuilt `v0.10.0` server archives, Claude `.mcpb` bundles, Python `cp39-abi3` `.whl` wheels, and TypeScript `openie-jouledb-0.10.0.tgz` ship on GitHub Releases (`GLIBC <= 2.35` on Linux); client SDKs are not yet published to PyPI, npm, or NuGet.
