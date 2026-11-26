# JouleDB: An Energy-Metered Multi-Model Database & Sub-1-Bit Weights Research Substrate

## Architecture & Empirical Status Paper — Version `0.8.0` (Pre-1.0 Research Preview, November 2026)
**Open Interface Engineering, Inc. (`OpenIE`)**  
**Website:** [https://jouledb.org](https://jouledb.org) | **Docs Book:** [https://openie-dev.github.io/jouledb](https://openie-dev.github.io/jouledb) | **GitHub:** [openIE-dev/jouledb](https://github.com/openIE-dev/jouledb) | **Hugging Face (`.jsyn` Embedder & Raw Benchmarks):** [huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)  
**License:** `Apache-2.0 OR MIT`

> **Revision & Audit Errata Note (`v0.8.0`, 2026-11-25):** Following our `v0.6.0` external-audit remediation (which retracted the premature `v1.0.0` GA tag, removed the unrelated `1.11M`-line `joule-web` crate, deleted synthetic frontier AI exam endpoints, and retracted 81 unverified low-BPW causal LLM exports from Hugging Face) and `v0.7.0` Raft `ReadIndex` / MVCC fixes, **`v0.8.0`** describes the **`41`-crate (`732,508` LOC, `12,269` tests)** codebase with **7** `deny(clippy::unwrap_used)` crates, out-of-core `.jcol` segment execution completing **`43 / 43` Official ClickBench queries on the full `99,997,497`-row (`14.78 GB`) dataset in `6.20 GB` post-load RSS (`20.74 GB` kernel `VmHWM` peak)**, head-to-head `100`-warehouse TCP `pgwire` OLTP (`1,659.5 -> 3,353.5 TPS`, `invariant_verified = true`) benchmarked directly against `PostgreSQL 16.16` (`fsync=on`) and `SQLite 3.53.1`, and 5-node TCP Raft §6.4 **`ReadIndex`** linearizable reads verified by official Clojure `elle 0.2.8`.

---

## Abstract

Modern data and AI stacks partition state across separate OLTP databases, analytical engines, vector indexes, container sandboxes, and neural inference runtimes. **JouleDB (`v0.8.0`)** explores unifying these workloads inside a single open-source Rust workspace where Copy-on-Write (`CoW`) Serializable Snapshot Isolation (`SSI`) tables, out-of-core `.jcol` vectorized columnar scans, PostgreSQL `pgwire` SQL, 5-node TCP Raft `ReadIndex` consensus, `10,240`-bit Hyperdimensional Computing (`Amorphic`) holograms, and sub-1-bit factorized sentence-embedding bitplanes (`.jsyn`) share one storage and energy-accounting substrate. Every energy figure is tagged by measurement provenance: **`reported_j`** from bare-metal OS hardware counters (Linux RAPL, macOS IOKit SMC) over multi-millisecond windows, or **`est_j`** from analytical cost models for sub-millisecond operations.

---

## 1. Core Subsystems in `v0.8.0` (`41` Workspace Crates)

1. **Storage, `SSI` Concurrency & Group-Committed Recovery (`joule-db-core`, `joule-db-local`):**
   - Lock-free positional `pread`/`pwrite`, `IoRingEngine` extent coalescing, Cahill `SSI` (`rw`-antidependency cycle prevention), index-assisted `MvccTableStorage` with `O(1)` PK/UNIQUE index checks, `commit_with_hook` + `GroupSync` shared `fsync`, `CRC-32` torn-WAL truncation, LSN-ordered replay, `ALP + FSST` columnar codecs, `RaBitQ` 1-bit `POPCNT` vector quantization, and double-entry `i128` ledger primitives (`tiger_ledger`).
   - **Compiler Safety Scope:** `#![deny(unsafe_code)]` in `joule-db-core` with **1** `#![allow(unsafe_code)]` module (`crates/joule-db-core/src/storage/mmap_extent.rs`); `#![cfg_attr(not(test), deny(clippy::unwrap_used))]` enforced in **7** crates (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`, `joule-db-local`, `joule-db-energy`, `joule-db-weights`, `joule-db-query`, plus `#![deny(clippy::unwrap_used)]` on `crates/joule-db-server/src/disk_col_segments.rs`), with `warn` across the remaining workspace crates.
2. **Out-of-Core `.jcol` Vectorized Execution, `pgwire` & Incremental Views (`joule-db-query`, `joule-db-server`):**
   - PostgreSQL v3 wire server (`pgwire` on `port + 2`, Simple & Extended Pipeline modes with §55.2.3 `Sync` recovery), bounded-chunk streaming Parquet `COPY` spilling `262,144`-row `.jcol` disk segments (`I8/I16/I32/I64` narrowing, bitpacked null masks, compact `u8/u16/u32` dictionaries, CRC-32 slice + `JCL1` footer checksums), `execute_segmented` (`pread` column-subset loading, two-phase `WHERE`-first filtering, 64-shard Radix `GROUP BY`, `VCol` virtual expressions, algebraic `SUM(x+k)` sharing, two-phase Top-K scans), `DBSP` `ZSet<T>` incremental view maintenance, `Arrow Flight SQL` (`JDB_IPC1`), and experimental Cypher, Datalog, SPARQL, Gremlin, and `SigQL` biosignal frontends.
3. **5-Node Real TCP Raft Consensus with `ReadIndex` (`raft.rs`, `raft_server.rs`, `raft_transport.rs`):**
   - Quorum write replication via `RaftNode::propose_and_commit` (Raft §5.4.2 current-term safety), quorum linearizable reads via `RaftNode::read_linearizable` (Raft §6.4 `ReadIndex` heartbeat quorum with zero WAL log growth), `CRC-32` durable `raft.wal` persistence, and 2PC participant/coordinator recovery logs.
4. **`.jsyn` (`JSYN-v1`) Sub-1-Bit Weights & `Amorphic` Continuum (`joule-db-weights`, `joule-db-amorphic`, `joule-db-gpu`):**
   - Compiles `.safetensors` and `.gguf` checkpoints into packed `u64` sign bitplanes (`LittleBit-2 ⊕ Sakana SVF`) executed via CPU `T-MAC` byte lookup tables or `wgpu` `countOneBits` shaders (`0` PyTorch / `llama.cpp` / ONNX Runtime).
   - Dissolves heterogeneous JSON, CSV, XML, `k=v`, prose logs, and N-Triples into `10,240`-bit Kanerva VSA holograms (`StreamingAmorphicContinuum` + `MorphRule` SI unit normalization).
5. **Confinement, Branching & Energy Ledger (`joule-db-branch`, `joule-db-energy`, `joule-db-ledger`):**
   - $O(1)$ CoW workspace/table branching (`JouleAgentFS`, `JouleCapsule` daemonless OCI v2 layer CAS), `HMAC-SHA256` `ThermoCap` capability tokens, technical security-control primitives (`trust_compliance.rs`), and Merkle-anchored energy receipts (`reported_j` / `est_j`).

---

## 2. Reproducible External Evaluations (`bench/`)

### 2.1 Full Official ClickBench (`hits.parquet`, `99,997,497` Rows, `105` Columns, `43` Queries) vs. DuckDB `2.0.1`

Evaluated via [`bench/clickbench_official.py`](../bench/clickbench_official.py) on `framework-128gb-0000` (`AMD Ryzen AI Max+ 395`, 32 threads, `128 GB` RAM, Linux `rapl-package` **`reported_j`**) over real `pgwire` (`psycopg 3.3.9`):

| Official `14.78 GB` `hits.parquet` Scale | Load (`DuckDB` vs `JouleDB`) | JouleDB RSS (`VmHWM` Peak) | Queries Succeeded | SQL-Valid Match (`Exact`) | Faster (`JouleDB` / `DuckDB`) | Geomean Latency (`JouleDB / DuckDB`) | Geomean Energy (`reported_j`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`100,000` rows** | `0.197 s` vs `0.355 s` | `933.8 MB` | **`43 / 43`** (`0` err) | **`43 / 43`** (`35` exact) | **`25`** / `18` | **`1.30×` slower** (**`median 0.80×`**) | **`1.53×` more J** |
| **`1,000,000` rows** | `0.147 s` vs `0.965 s` | `1,236.9 MB` (`2,196.6 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`28` exact) | **`8`** / `35` | **`3.01×` slower** (`median 3.14×`) | **`3.07×` more J** |
| **`10,000,000` rows** | `1.402 s` vs `10.755 s` | `2,405.9 MB` (`9,006.2 MB`) | **`43 / 43`** (`0` err) | **`43 / 43`** (`29` exact) | **`10`** / `33` | **`3.21×` slower** (`median 4.73×`) | **`3.21×` more J** |
| **`99,997,497` rows (FULL `100M`)** | `15.546 s` vs `91.621 s` | **`6,199.0 MB` (`20,740.5 MB`)** | **`43 / 43`** (`0` err) | **`43 / 43`** (`31` exact) | **`5`** / `38` | **`4.85×` slower** (`median 4.57×`) | **`4.25×` more J** |

### 2.2 Head-to-Head `100`-Warehouse TCP `pgwire` OLTP (`100,000` Accounts) vs. `PostgreSQL 16.16` & `SQLite 3.53.1`

Evaluated via [`bench/oltp_pgwire.py`](../bench/oltp_pgwire.py) ([`oltp_pgwire_100wh.json`](../bench/results/oltp_pgwire_100wh.json)) across `1, 4, 8, 16` concurrent `psycopg` TCP clients (`50%` indexed point read, `30%` durable `fsync=on` `BEGIN..UPDATE..INSERT..COMMIT`, `20%` warehouse aggregate; `invariant_verified = true` across all 5 runs):
- **JouleDB `0.8.0` (`pgwire` Simple Query, 4-RTT + `fsync`):** **`1,659.49 TPS`** (`c=1`, `p50 0.304 ms`, `21.15 mJ/txn`, `0` errors — **`1.44×` faster than `PostgreSQL 16.16` at `c=1`**) -> `1,086.43 TPS` (`c=4`) -> `2,397.52 TPS` (`c=8`) -> **`3,353.47 TPS`** (`c=16`, `p50 1.076 ms`, `17.23 mJ/txn`, `2` SI conflicts; `20,366` committed writes, `SUM(balance) == 100,289,630`).
- **PostgreSQL `16.16` (`pgwire` Simple Query & Extended Pipeline, `fsync=on`, `synchronous_commit=on`):** `1,149.05 -> 11,452.46 TPS` (`p50 0.077–0.253 ms`, `3.79 mJ/txn` at `c=16`, `51,319` writes) in 4-RTT Simple Query mode, and `1,255.37 -> 10,441.83 TPS` (`3.71 mJ/txn`, `49,555` writes) in 1-RTT Pipeline mode.
- **JouleDB `0.8.0` (`pgwire` Extended Pipeline, 1-RTT + `fsync`):** `256.92 -> 593.70 TPS` (`4,394` writes, `invariant_verified = true`) — slower than Simple Query today because JouleDB's Extended Query `Describe` (`'D'`) handler executes a dry-run schema probe before `Execute`.
- **Embedded `SQLite 3.53.1` (`WAL`, `synchronous=NORMAL`, in-process C API):** `43,264.1 -> 111,119.3 TPS` (`0.36 mJ/txn`, no TCP wire hops or per-commit `fsync`).

### 2.3 5-Node Real TCP Raft (`ReadIndex`) Fault-Injection + Official Clojure `elle 0.2.8`

Evaluated via [`crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs`](../crates/joule-db-server/tests/raft_tcp_elle_fault_injection.rs) and [`bench/elle_check.clj`](../bench/elle_check.clj):
- **Faults Injected:** 5 `RaftNode<KvStateMachine, TcpRaftTransport>` servers across a 20-link socket-severing `TcpPartitionMatrix`; asymmetric `2/5` vs `3/5` split isolating initial leader -> minority write & `ReadIndex` (`read_linearizable`) rejection -> `3/5` higher-term leader election -> follower stop + disk `raft.wal` restart -> partition heal & 5-node convergence (`x = [17; 5]`, `y = [18; 5]`, `lost_committed_writes = 0`).
- **Official `elle 0.2.8` Verdict:** **`{:valid? true}`** (`rw-register`, `:strict-serializable`, `76` events).

### 2.4 `.jsyn` Compression vs. Fidelity Boundaries

- **Verified High-Fidelity Encoder (`0.4378 BPW`):** [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`Apache-2.0`, `1.95 MB`, `22.23×` smaller than `FP16`) achieves **`0.9441` mean sentence-embedding cosine similarity** against upstream `FP32`.
- **Degraded Ultra-Low-BPW Causal LLMs (`0.02–0.17 BPW`, Retracted):** Out-of-sample rank sweeps (`r = 8..128`) on `Gemma-4-E2B` (`q_proj`) and `Inkling-2-Mini` (`128`-Exp MoE) yield **`0.113–0.455` output cosine** (`KL = 2.59–3.87`, `65.6%` Top-5 agreement at `r = 128`). All 81 unverified causal LLM checkpoints remain deleted from Hugging Face.

---

## 3. Explicit Engineering & Governance Boundaries

1. **Pre-1.0 Research Preview (`v0.8.0`):** Out-of-core `.jcol` segments (`6.20 GB` load RSS, `20.74 GB` peak `VmHWM`) resolve the `v0.7.0` `100M`-row memory ceiling and execute all `43 / 43` Official ClickBench queries cleanly, while mature specialized engines (`DuckDB 2.0.1` `4.85×` faster on `100M` geomean; `PostgreSQL 16.16` `3.42×` faster at `c=16` `pgwire` OLTP; `SQLite` embedded WAL) remain ahead on single-model OLAP/OLTP throughput.
2. **No Third-Party Compliance Certifications:** `trust_compliance.rs` implements technical controls (SHA-256 hash-chained audit logs, WORM retention locks, per-subject DEK zeroization, Luhn redaction, `CycloneDX v1.6` SBOM, Bell-LaPadula lattice labels, and unvalidated Rust `ML-KEM-768`/`ML-DSA-65` signatures). **No SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been performed.**
3. **In-Tree SDKs & Portable Binaries:** Prebuilt `v0.8.0` server binaries target macOS (`aarch64`/`x86_64`) and Linux (`GLIBC <= 2.35`, verified on Ubuntu 24.04 LTS & Debian 12). Client SDKs (`sdks/*`, `python/`) live in-tree in this repository and are not yet published to PyPI, npm, or NuGet.
