# JouleDB: An Energy-Metered Multi-Model Database & Sub-1-Bit Weights Research Substrate

## Architecture & Empirical Status Paper — Version `0.9.0` (Pre-1.0 Research Preview, November 2026)
**Open Interface Engineering, Inc. (`OpenIE`)**  
**Website:** [https://jouledb.org](https://jouledb.org) | **Docs Book:** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb) | **GitHub:** [openIE-dev/jouledb](https://github.com/openIE-dev/jouledb) | **Hugging Face (`.jsyn` Embedder & Raw Benchmarks):** [huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)  
**License:** `Apache-2.0 OR MIT`

> **Revision & Audit Errata Note (`v0.9.0`, 2026-11-26):** Following our `v0.6.0` external-audit remediation (which retracted the premature `v1.0.0` GA tag, removed the unrelated `1.11M`-line `joule-web` crate, deleted synthetic frontier AI exam endpoints, and retracted 81 unverified low-BPW causal LLM exports from Hugging Face) and `v0.7.0`–`v0.8.0` storage/analytical upgrades, **`v0.9.0`** describes the **`41`-crate (`735,187` LOC, `12,272` tests)** codebase with **7** `deny(clippy::unwrap_used)` crates, built-in **Claude/MCP Apps (`SEP-1865`) Studio UI (`ui://jouledb/studio.html`, `50` tools, `.mcpb` bundles)**, accelerated out-of-core `.jcol` segment execution completing **`43 / 43` Official ClickBench queries on the full `99,997,497`-row (`14.78 GB`) dataset in `6.20 GB` post-load RSS (`15.48 GB` kernel `VmHWM` peak, `2.82×` geomean latency & `2.43×` `reported_j` vs DuckDB `2.0.1`, `8 / 43` wins)**, `100`-warehouse TCP `pgwire` OLTP (`3,351.0 TPS` Simple, `1,790.0 TPS` Extended Pipeline, `invariant_verified = true`) benchmarked against `PostgreSQL 16.16` (`fsync=on`) and `SQLite 3.53.1`, and **5-node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex` + `SIGKILL` + `raft.wal`)** verified **`{:valid? true}`** by official Clojure `elle 0.2.8`.

---

## Abstract

Modern data and AI stacks partition state across separate OLTP databases, analytical engines, vector indexes, agent UI sidecars, container sandboxes, and neural inference runtimes. **JouleDB (`v0.9.0`)** explores unifying these workloads inside a single open-source Rust workspace where Copy-on-Write (`CoW`) Serializable Snapshot Isolation (`SSI`) tables, out-of-core `.jcol` vectorized columnar scans, PostgreSQL `pgwire` SQL, multi-host WAN TCP Raft `ReadIndex` consensus, `SEP-1865` MCP Apps interactive UI, `10,240`-bit Hyperdimensional Computing (`Amorphic`) holograms, and sub-1-bit factorized sentence-embedding bitplanes (`.jsyn`) share one storage and energy-accounting substrate. Every energy figure is tagged by measurement provenance: **`reported_j`** from bare-metal OS hardware counters (Linux RAPL, macOS IOKit SMC) over multi-millisecond windows, or **`est_j`** from analytical cost models for sub-millisecond operations.

---

## 1. Core Subsystems in `v0.9.0` (`41` Workspace Crates)

1. **Storage, `SSI` Concurrency & Group-Committed Recovery (`joule-db-core`, `joule-db-local`):**
   - Lock-free positional `pread`/`pwrite`, `IoRingEngine` extent coalescing, Cahill `SSI` (`rw`-antidependency cycle prevention), index-assisted `MvccTableStorage` with `O(1)` PK/UNIQUE index checks, `commit_with_hook` + `GroupSync` shared `fsync`, `CRC-32` torn-WAL truncation, LSN-ordered replay, `ALP + FSST` columnar codecs, `RaBitQ` 1-bit `POPCNT` vector quantization, and double-entry `i128` ledger primitives (`tiger_ledger`).
   - **Compiler Safety Scope:** `#![deny(unsafe_code)]` in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`); `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` enforced in **7** crates (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`, `joule-db-local`, `joule-db-energy`, `joule-db-weights`, `joule-db-query`, plus `#![deny(clippy::unwrap_used)]` on `disk_col_segments.rs` and `mcp_app_ui.rs`), with `warn` across the remaining workspace crates.
2. **Out-of-Core `.jcol` Vectorized Execution, `pgwire` & Incremental Views (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire server (`pgwire` on `port + 2`, Simple & static-`Describe` Extended Pipeline modes with §55.2.3 `Sync` recovery), row-budgeted (`MAX_CHUNK_ROWS = 524,288`) streaming Parquet `COPY` spilling `262,144`-row `.jcol` disk segments (`I8/I16/I32/I64` narrowing, `global_int_sum` + ZoneMap headers, bitpacked null masks, compact `u8/u16/u32` dictionaries, CRC-32 slice + `JCL1` footer checksums), `execute_segmented` (zero-I/O metadata `COUNT`/`MIN`/`MAX`/`SUM`/`AVG`, `pread` column-subset loading, dynamic `AtomicI64` Top-K ZoneMap pruning, zero-`Arc<str>` `RawTextSlice` text distinct/grouping, 256-partition `CompactI2Row` in-place sort-reduce), and `DBSP` `ZSet<T>` incremental view maintenance.
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
| **`100,000` rows** | `0.326 s` vs `0.409 s` | `927.5 MB` (`2,719.5 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`35` exact) | **`21`** / `22` | **`1.31×` slower** (`median 1.04×`) | **`1.65×` more J** |
| **`1,000,000` rows** | `0.257 s` vs `1.363 s` | `1,246.4 MB` (`1,958.6 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`13`** / `30` | **`2.42×` slower** (`median 3.22×`) | **`2.34×` more J** |
| **`10,000,000` rows** | `1.459 s` vs `12.568 s` | `2,163.3 MB` (`9,060.4 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`12`** / `31` | **`2.52×` slower** (`median 3.88×`) | **`2.47×` more J** |
| **`99,997,497` rows (FULL `100M`)** | `16.497 s` vs `86.068 s` | **`6,204.0 MB` (`15,483.2 MB`)** | **`43 / 43`** (`0` err) | **`43 / 43`** (`31` exact) | **`8`** / `35` | **`2.82×` slower** (`median 3.42×`) | **`2.43×` more J** |

### 2.2 Head-to-Head `100`-Warehouse TCP `pgwire` OLTP (`100,000` Accounts) vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

Evaluated via [`bench/oltp_pgwire.py`](../bench/oltp_pgwire.py) ([`oltp_pgwire_100wh.json`](../bench/results/oltp_pgwire_100wh.json)) across `1, 4, 8, 16` concurrent `psycopg` TCP clients (`50%` indexed point read, `30%` durable `fsync=on` `BEGIN..UPDATE..INSERT..COMMIT`, `20%` warehouse aggregate; `invariant_verified = true` across all 5 runs):
- **JouleDB `0.9.0` (`pgwire` Simple Query, 4-RTT + `fsync`):** `522.23 TPS` (`c=1`, `p50 0.340 ms`) -> `954.49 TPS` (`c=4`) -> `2,125.09 TPS` (`c=8`) -> **`3,351.04 TPS`** (`c=16`, `p50 1.069 ms`, `24.29 mJ/txn`, `0` errors, `16,647` committed writes, `SUM(balance) == 100,238,570`).
- **JouleDB `0.9.0` (`pgwire` Extended Pipeline, 1-RTT + `fsync`):** `260.01 TPS` (`c=1`) -> `871.90 TPS` (`c=4`) -> **`1,789.98 TPS`** (`c=8`) -> **`1,703.25 TPS`** (`c=16`, `10,973` writes, `invariant_verified = true` — **`3.01×` higher peak TPS & `2.50×` more writes than `v0.8.0`'s `593.70 TPS`** via static `describe_sql_static` OID inference).
- **PostgreSQL `16.16` (`pgwire` Simple & Extended Pipeline, `fsync=on`, `synchronous_commit=on`):** `3,504.95 -> 10,815.93 TPS` (`9.13 mJ/txn` at `c=16`, `55,902` writes) in Simple Query mode, and `1,291.43 -> 8,634.44 TPS` (`14.66 mJ/txn`, `43,602` writes) in Pipeline mode.
- **Embedded `SQLite 3.53.1` (`WAL`, `synchronous=NORMAL`, in-process C API):** `33,431.06 -> 60,263.42 TPS` (`2.18 mJ/txn`, `506,957` writes, no TCP wire hops or per-commit `fsync`).

### 2.3 5-Node Multi-Host Cross-Machine WAN/Tailscale TCP Raft (`ReadIndex`) + Official Clojure `elle 0.2.8`

Evaluated via [`bench/raft_multihost_wan.py`](../bench/raft_multihost_wan.py) ([`raft_wan_multihost_summary.json`](../bench/results/raft_wan_multihost_summary.json)) and [`bench/elle_check.clj`](../bench/elle_check.clj):
- **Cross-Machine WAN Faults:** 5 separate OS `raft_wan_node` processes across **2 distinct physical Linux hosts** (`framework-128gb-0000` bare-metal + `openie-core-1` Cloud Debian 12 VM, **`27.58 ms` median Tailscale WAN TCP RTT**); cross-WAN `2/5` minority split (`["n1", "n4"]` vs `["n2", "n3", "n5"]`) isolating initial leader `n1` -> stale write & `ReadIndex` rejection -> `3/5` majority elects `n3` (term `2`, `24` ops) -> real OS `SIGKILL` (`kill -9`) on `n5` (`16` ops during crash) -> `raft.wal` restart & 5-node convergence (`commit=68, applied=68, x=68, y=39`).
- **Official `elle 0.2.8` Verdict:** **`{:valid? true}`** (`rw-register`, `:strict-serializable`, `270` WAN events; local `tests/raft_tcp_elle_fault_injection.rs` `76` events also `{:valid? true}`).

### 2.4 `.jsyn` Compression vs. Fidelity Boundaries

- **Verified High-Fidelity Encoder (`0.4378 BPW`):** [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`Apache-2.0`, `1.95 MB`, `22.23×` smaller than `FP16`) achieves **`0.9441` mean sentence-embedding cosine similarity** against upstream `FP32`.
- **Degraded Ultra-Low-BPW Causal LLMs (`0.02–0.17 BPW`, Retracted):** Out-of-sample rank sweeps (`r = 8..128`) on `Gemma-4-E2B` (`q_proj`) and `Inkling-2-Mini` (`128`-Exp MoE) yield **`0.113–0.455` output cosine** (`KL = 2.59–3.87`, `65.6%` Top-5 agreement at `r = 128`). All 81 unverified causal LLM checkpoints remain deleted from Hugging Face.

---

## 3. Explicit Engineering & Governance Boundaries

1. **Pre-1.0 Research Preview (`v0.9.0`):** Out-of-core `.jcol` segments (`6.20 GB` load RSS, `15.48 GB` peak `VmHWM`, `2.82×` geomean vs DuckDB `2.0.1` on `100M` rows) and static-`Describe` `pgwire` pipelining (`1,790 TPS` Pipelined, `3,351 TPS` Simple) narrow the gap substantially, while mature specialized engines (`DuckDB 2.0.1` `2.82×` faster on `100M` geomean; `PostgreSQL 16.16` `3.23×` faster at `c=16` `pgwire` OLTP; `SQLite` embedded WAL) remain ahead on single-model OLAP/OLTP throughput.
2. **No Third-Party Compliance Certifications:** `trust_compliance.rs` implements technical controls (`external_audit_performed = false`). **No SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been performed.**
3. **SDK Artifacts & Portable Binaries:** Prebuilt `v0.9.0` server archives, Claude `.mcpb` bundles, Python `cp39-abi3` `.whl` wheels, and TypeScript `openie-jouledb-0.9.0.tgz` ship on GitHub Releases (`GLIBC <= 2.35` on Linux); client SDKs are not yet published to PyPI, npm, or NuGet.
